# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

def extract_bbox(response):
    start_tag = "<answer>"
    end_tag = "</answer>"
    input_str = response
    # Check if the start tag is in the string
    if start_tag in input_str:
        # Extract the content between the start tag and end tag
        start_idx = input_str.find(start_tag) + len(start_tag)
        end_idx = input_str.find(end_tag)
        
        # If end_tag is not found (i.e., the string is truncated), assume it should be at the end
        if end_idx == -1:
            end_idx = len(input_str)
    
        content_str = input_str[start_idx:end_idx]
    
        # Check if it ends with a closing bracket, if not, fix it
        if not content_str.endswith("]"):
            # If the string is truncated, remove the incomplete part
            content_str = content_str.rsplit("},", 1)[0] + "}]"
    
        # Replace single quotes with double quotes for valid JSON
        content_str_corrected = content_str.replace("'", '"')
    
        # Convert the corrected string to a list of dictionaries (JSON format)
        try:
            bbox_list = json.loads(content_str_corrected)
        except json.JSONDecodeError as e:
            bbox_list = None
    else:
        bbox_list = None
    return bbox_list

def calculate_iou(bbox1, bbox2):
    # 获取两个bbox的坐标
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集区域的左上角和右下角
    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    # 如果没有交集，返回0
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    # 计算交集区域的面积
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
    # 计算两个bbox的面积
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算并集区域的面积
    union_area = area1 + area2 - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area
    return iou

def sort_and_calculate_iou(list1, list2, iou_threshold=0.5):
    # 按照confidence从大到小排序
    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    # 用于存储IoU结果
    iou_results = []
    
    # 用于存储匹配后的bbox
    matched_list1_indices = set()  # 记录已经匹配的list1的index

    # 遍历排序后的list2中的每个bbox
    for bbox2 in list2_sorted:
        max_iou = 0
        matched_bbox1 = -1  # 当前list2中的bbox与哪个list1中的bbox匹配
        best_iou = 0
        
        # 对每个未匹配的list1中的bbox进行匹配
        for i, bbox1 in enumerate(list1):
            if i not in matched_list1_indices:  # 确保list1中该bbox没有被匹配过
                iou = calculate_iou(bbox1['Position'], bbox2['Position'])
                if iou > best_iou:
                    best_iou = iou
                    matched_bbox1 = i
        
        # 如果找到了最佳匹配（IoU大于阈值）
        if best_iou > iou_threshold:
            iou_results.append((best_iou, bbox2['Confidence']))  # 记录IoU和confidence
            matched_list1_indices.add(matched_bbox1)  # 标记list1中的bbox为已匹配
        else:
            # 如果没有找到匹配的bbox，IoU为0
            iou_results.append((0, bbox2['Confidence']))
    
    ### [(0.7192676547515258, 1.0), (0, 0.7)]
    return iou_results

def remove_duplicates(bbox_list):
    seen = set()
    unique_bboxes = []
    
    for bbox in bbox_list:
        # Convert the position tuple to a tuple for set hashing
        position_tuple = tuple(bbox['Position'])
        
        if position_tuple not in seen:
            seen.add(position_tuple)
            unique_bboxes.append(bbox)
    
    return unique_bboxes

### 老版本的 reward 计算方法
def compute_reward(iou_results):
    reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        if temp_iou!=0: ### 匹配成功
            total_reward = (temp_iou + (temp_iou + temp_confidence - 1) ** 2)/2
        elif temp_iou==0: ### 匹配失败
            total_reward = 0
        reward += total_reward
        
    reward = reward/len(iou_results)
    return reward

### 新版本的 reward 计算方法
def compute_reward_iou(iou_results):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    iou_reward = iou_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return iou_reward

### 新版本的 reward 计算方法 ，改进IoU reward计算
def compute_reward_iou_v2(iou_results, len_gt):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    if len_gt>=len(iou_results):
        iou_reward = iou_reward/len_gt
    else:
        iou_reward = iou_reward/len(iou_results)
    return iou_reward

def compute_reward_confidence(iou_results):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    iou_reward = iou_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return confidence_reward

def accuracy_reward_iou(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        student_answer_bbox = []
        ground_truth_bbox = []
        iou_results = []
        show_flage = 0

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                show_flage = 1
                ##########################################
                # # Extract answer from solution if it has think/answer tags
                # sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                # ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # # Extract answer from content if it has think/answer tags
                # content_match = re.search(r'<answer>(.*?)</answer>', content)
                # student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # # Compare the extracted answers
                # if student_answer == ground_truth:
                #     reward = 1.0
                ##########################################
                # Extract answer from solution if it has think/answer tags
                ground_truth = sol.strip()
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = '<answer>'+student_answer+'</answer>'

                ### 获得bbox和confidence
                student_answer = student_answer.replace("[[",'[')  ### 修正student_answer中的格式错误
                student_answer = student_answer.replace("]]",']')  ### 修正student_answer中的格式错误
                student_answer = student_answer.replace("\n",'')  ### 修正student_answer中的格式错误
                ### [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
                # pdb.set_trace()
                ground_truth_bbox = extract_bbox(ground_truth)   ### 提取bbox和confidence
                student_answer_bbox = extract_bbox(student_answer)  ### 提取bbox和confidence
                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:  ### 没有提取出正确的bbox
                    reward = 0.0
                else:
                    # import pdb; pdb.set_trace()
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   ### 模型有时候会复读，去除重复的bbox
                    # pdb.set_trace()
                    # 计算IoU，如果bbox数量不同，取最小的bbox数量
                    iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
                    # pdb.set_trace()
                    ### v1 所有预测的bbox的IoU的和除以预测的bbox的数量，可能导致模型喜欢只出一个bbox
                    # reward = compute_reward_iou(iou_results)
                    ### v2 所有预测的bbox的IoU的和除以ground truth的bbox的数量
                    reward = compute_reward_iou_v2(iou_results, len(ground_truth_bbox))
                    if reward>1:
                        reward = 1.0
                    # import pdb; pdb.set_trace()

                if 'No Object' in student_answer and 'No Object' in ground_truth:  ### 物体不出现在图片中，需要拒绝回答
                    reward = 1.0
                ##########################################
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of IoU: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if show_flage==1:
                    f.write(f"student_answer_bbox: {student_answer_bbox}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                    if student_answer_bbox!=None:
                        f.write(f"iou_results: {iou_results}\n")
        show_flage = 0 
    return rewards

def accuracy_reward_confidence(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        student_answer_bbox = []
        ground_truth_bbox = []
        iou_results = []
        show_flage = 0

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                show_flage = 1
                ##########################################
                # # Extract answer from solution if it has think/answer tags
                # sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                # ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # # Extract answer from content if it has think/answer tags
                # content_match = re.search(r'<answer>(.*?)</answer>', content)
                # student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # # Compare the extracted answers
                # if student_answer == ground_truth:
                #     reward = 1.0
                ##########################################
                # Extract answer from solution if it has think/answer tags
                ground_truth = sol.strip()
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = '<answer>'+student_answer+'</answer>'

                ### 获得bbox和confidence
                student_answer = student_answer.replace("[[",'[')  ### 修正student_answer中的格式错误
                student_answer = student_answer.replace("]]",']')  ### 修正student_answer中的格式错误
                student_answer = student_answer.replace("\n",'')  ### 修正student_answer中的格式错误
                ### [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
                # pdb.set_trace()
                ground_truth_bbox = extract_bbox(ground_truth)   ### 提取bbox和confidence
                student_answer_bbox = extract_bbox(student_answer)  ### 提取bbox和confidence
                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:  ### 没有提取出正确的bbox
                    reward = 0.0
                else:
                    # import pdb; pdb.set_trace()
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   ### 模型有时候会复读，去除重复的bbox
                    # pdb.set_trace()
                    # 计算IoU，如果bbox数量不同，取最小的bbox数量
                    iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
                    # pdb.set_trace()
                    reward = compute_reward_confidence(iou_results)
                    if reward>1:
                        reward = 1.0
                    if reward<0:
                        reward = 0.0
                    # import pdb; pdb.set_trace()

                if 'No Object' in student_answer and 'No Object' in ground_truth:  ### 物体不出现在图片中，需要拒绝回答
                    reward = 1.0
                ##########################################
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of Confidence: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if show_flage==1:
                    f.write(f"student_answer_bbox: {student_answer_bbox}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                    if student_answer_bbox!=None:
                        f.write(f"iou_results: {iou_results}\n")
        show_flage = 0 
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

# ### 老的 reward registry，分为两部分
# reward_funcs_registry = {
#     "accuracy": accuracy_reward,
#     "format": format_reward,
# }

### 新的 reward registry，分为三部分
reward_funcs_registry = {
    "accuracy_iou": accuracy_reward_iou,
    "accuracy_confidence": accuracy_reward_confidence,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    # import pdb; pdb.set_trace()
    script_args.reward_funcs = ['accuracy_iou','accuracy_confidence','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # import pdb; pdb.set_trace()

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    ### lzy modified
    from datasets import DatasetDict
    dataset = DatasetDict.load_from_disk(script_args.dataset_name)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    # QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
    #                 ],
    #             },
    #         ],
    #     }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
