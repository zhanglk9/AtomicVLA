"""
Video processor using InternVideo for segment-level analysis.

- Segment loading and preprocessing
- InternVideo inference
- Action description parsing
"""

import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


class VideoProcessor:
    """InternVideo video processor."""

    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVideo2_5_Chat_8B",
        device_id: int = 0,
        task_mapping: Optional[Dict[str, List[str]]] = None,
    ):
        if torch.cuda.is_available():
            self.device = f"cuda:{device_id}"
            torch.cuda.set_device(device_id)
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
        )

        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.task_mapping = task_mapping or {}

        self.generation_config = dict(
            do_sample=True,
            temperature=0.1,
            max_new_tokens=300,
            top_p=0.9,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def build_transform(self, input_size: int = 448):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])

    def dynamic_preprocess(
        self,
        image,
        min_num: int = 1,
        max_num: int = 6,
        image_size: int = 448,
        use_thumbnail: bool = False,
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(target_aspect_ratio[0] * target_aspect_ratio[1]):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _normalize_patches(
        self, pixel_values_list, num_patches_list, target_patches=None
    ):
        if not pixel_values_list:
            return None, []
        if target_patches is None:
            target_patches = min(num_patches_list)
        normalized_pixel_values = []
        normalized_num_patches = []
        for pixel_values, num_patches in zip(pixel_values_list, num_patches_list):
            if num_patches == target_patches:
                normalized_pixel_values.append(pixel_values)
                normalized_num_patches.append(num_patches)
            elif num_patches > target_patches:
                normalized_pixel_values.append(pixel_values[:target_patches])
                normalized_num_patches.append(target_patches)
            else:
                last_patch = pixel_values[-1].unsqueeze(0)
                padding_needed = target_patches - num_patches
                padding = last_patch.repeat(padding_needed, 1, 1, 1)
                padded = torch.cat([pixel_values, padding], dim=0)
                normalized_pixel_values.append(padded)
                normalized_num_patches.append(target_patches)
        shapes = [pv.shape for pv in normalized_pixel_values]
        if len(set(shapes)) > 1:
            min_shape = min(shapes)
            normalized_pixel_values = [pv[:min_shape[0]] for pv in normalized_pixel_values]
            normalized_num_patches = [min_shape[0]] * len(normalized_pixel_values)
        return normalized_pixel_values, normalized_num_patches

    def _convert_av1_to_h264(self, video_path: str) -> Optional[str]:
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_path = temp_file.name
            temp_file.close()
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", temp_path,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=300,
            )
            if result.returncode == 0 and os.path.exists(temp_path):
                return temp_path
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
        except Exception:
            return None

    def load_video_segment(
        self,
        video_path: str,
        start_frame: float,
        end_frame: float,
        input_size: int = 448,
        max_num: int = 1,
        num_segments: int = 16,
    ) -> Tuple[Optional[torch.Tensor], Optional[List[int]]]:
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            if end_frame <= start_frame:
                return None, None

            half_frames = (end_frame - start_frame + 1) // 2
            if half_frames % 4 != 0:
                half_frames += 4 - (half_frames % 4)
            frame_indices = np.linspace(
                start_frame, end_frame - 1, half_frames, dtype=int
            )

            pixel_values_list, num_patches_list = [], []
            transform = self.build_transform(input_size=input_size)
            for frame_index in frame_indices:
                img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
                img = self.dynamic_preprocess(
                    img, image_size=input_size, use_thumbnail=True, max_num=max_num
                )
                pixel_values = torch.stack([transform(t) for t in img])
                num_patches_list.append(pixel_values.shape[0])
                pixel_values_list.append(pixel_values)

            if pixel_values_list:
                norm_pv, norm_np = self._normalize_patches(
                    pixel_values_list, num_patches_list
                )
                if norm_pv:
                    pixel_values = torch.cat(norm_pv).to(torch.bfloat16)
                    return pixel_values, norm_np
            return None, None

        except Exception:
            converted = self._convert_av1_to_h264(video_path)
            if not converted:
                return None, None
            try:
                vr = VideoReader(converted, ctx=cpu(0), num_threads=1)
                if end_frame <= start_frame:
                    return None, None
                frame_indices = np.linspace(
                    start_frame, end_frame - 1,
                    min(num_segments, int(end_frame - start_frame)),
                    dtype=int,
                )
                pixel_values_list, num_patches_list = [], []
                transform = self.build_transform(input_size=input_size)
                for frame_index in frame_indices:
                    img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
                    img = self.dynamic_preprocess(
                        img, image_size=input_size, use_thumbnail=True, max_num=max_num
                    )
                    pixel_values = torch.stack([transform(t) for t in img])
                    num_patches_list.append(pixel_values.shape[0])
                    pixel_values_list.append(pixel_values)
                if os.path.exists(converted):
                    os.remove(converted)
                if pixel_values_list:
                    norm_pv, norm_np = self._normalize_patches(
                        pixel_values_list, num_patches_list
                    )
                    if norm_pv:
                        return torch.cat(norm_pv).to(torch.bfloat16), norm_np
            except Exception:
                pass
            if os.path.exists(converted):
                os.remove(converted)
            return None, None

    def analyze_action_segment2(
        self,
        pixel_values,
        num_patches_list,
        segment_info: dict,
        task_instruct: str,
    ) -> Optional[str]:
        """Analyze the full clip and return a Task Chain string."""
        if pixel_values is None:
            return None
        expected = sum(num_patches_list)
        actual = pixel_values.shape[0]
        if expected != actual:
            raise ValueError(
                f"Tensor shape mismatch: expected {expected} patches, got {actual}"
            )
        pixel_values = pixel_values.to(self.device)
        video_prefix = "".join(
            f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))
        )
        prompt = self._build_cotv3_action_prompt_all(video_prefix, task_instruct)
        with torch.no_grad():
            action_description, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                self.generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
        return action_description

    def analyze_action_segment3(
        self,
        pixel_values,
        num_patches_list,
        history: str,
        segment_info: dict,
        task_instruct: str,
        previous_state: str,
        instruct: str,
    ) -> Tuple[Optional[dict], str]:
        """Parse the current step action from a Task Chain string."""
        result_list = None
        match = re.search(r"\[(.*?)\]", task_instruct)
        if match:
            content = match.group(1)
            result_list = [item.strip() for item in content.split(",")]

        if instruct in self.task_mapping:
            result_list = self.task_mapping[instruct]
            task_instruct = f"Task Chain: {result_list}"

        if not result_list:
            action_info = self._parse_cotv3_action_response(
                instruct, segment_info, instruct, task_instruct
            )
            return action_info, instruct

        steps = history.split("/")
        try:
            total_steps = int(steps[1])
            current_step = int(steps[0])
        except (IndexError, ValueError):
            action_info = self._parse_cotv3_action_response(
                instruct, segment_info, instruct, task_instruct
            )
            return action_info, instruct

        if len(result_list) != total_steps:
            if len(result_list) == 2 and total_steps == 4:
                result_list = (result_list * 2)[:4]
            else:
                action_info = self._parse_cotv3_action_response(
                    instruct, segment_info, instruct, task_instruct
                )
                return action_info, instruct

        action_description = result_list[current_step - 1]
        action_info = self._parse_cotv3_action_response(
            action_description, segment_info, instruct, task_instruct
        )
        if not self._validate_response_quality(action_info):
            raise ValueError(f"Model output validation failed: {action_info}")
        return action_info, action_description

    def _build_cotv3_action_prompt_all(
        self, video_prefix: str, task_instruct: str
    ) -> str:
        return f"""
{video_prefix}
You are analyzing a video segment of a robotic manipulation task: {task_instruct}. Please provide an analysis consisting of three parts based on the task name and observations:
1. Environment: Describe all object types present on the tabletop, and explicitly identify which objects are involved in interaction.
2. Instruct: {task_instruct}
3. Task Chain: List the sequence of concrete actions the robot must perform to complete the task, using concise verb-object phrases. Output in a single line, no line breaks.

It should be noted that if "move" appears, it generally needs to be broken down into two sub-steps: "pick up" and "place." Please provide the correct breakdown based on the video analysis.
If "open" appears in the instruction, it is generally divided into steps such as open/pick up/close.

Examples:
1. Task Chain: [Pick up the alphabet soup, place the alphabet soup into the basket, pick up the tomato sauce, place the tomato sauce into the basket.]
2. Task Chain: [Pick up the cup, adjust the robotic arm to place the cup, close the microwave door.]

Strictly adhere to this format. Finally, only output the **Task Chain** and its corresponding content.
"""

    def _parse_cotv3_action_response(
        self,
        response: str,
        segment_info: dict,
        instruct: str,
        task_instruct: str,
    ) -> dict:
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        action_name = lines[0] if lines else ""
        if action_name:
            action_name = self._format_action_name(action_name)
        else:
            action_name = f"Execute robotic manipulation {segment_info['step_number']}"
        action_verb = self._extract_action_verb(action_name)
        cot = (
            f"The {task_instruct}. This is step {segment_info['step_number']}"
            f"/{segment_info['all_step']} of task, {action_name}, "
            f"and the atomic skill is {action_verb}"
        )
        return {
            "action_name": instruct.strip(),
            "chain_of_thought": cot.strip(),
            "primary_action_verb": action_verb,
            "start_frame": segment_info["start_frame"],
            "end_frame": segment_info["end_frame"],
            "timestamp_formatted": (
                f"{segment_info['start_frame']} - {segment_info['end_frame']}"
            ),
        }

    def _format_action_name(self, raw_action: str) -> str:
        action = raw_action.strip().strip("'\"").strip(".")
        if len(action) > 120:
            truncated = action[:120]
            last_space = truncated.rfind(" ")
            action = action[: last_space + 1] + "..." if last_space > 80 else action[:120] + "..."
        if action and not action[0].isupper():
            action = action[0].upper() + action[1:]
        return action or "Execute robotic manipulation"

    def _extract_action_verb(self, text: str) -> str:
        action_verbs = [
            "pick", "place", "move", "scoop", "grasp", "transfer", "pour",
            "push", "pull", "rotate", "adjust", "insert", "close", "open",
            "make", "lift", "drop", "hold", "release", "slide", "twist", "turn",
            "pinch", "squeeze", "press", "tap", "touch", "brush", "wipe",
            "spread", "fold", "unfold", "roll", "unroll", "wrap", "unwrap",
            "assemble", "disassemble", "attach", "detach", "connect", "disconnect",
            "mount", "unmount", "install", "remove", "extract",
            "cut", "slice", "chop", "trim", "shape", "mold", "bend", "straighten",
            "measure", "weigh", "scan", "inspect", "examine", "check", "test",
            "clean", "wash", "dry", "polish", "scrub", "vacuum", "dust",
            "approach", "navigate", "position", "align", "orient", "center",
            "locate", "search", "track", "follow", "retreat", "advance",
            "fill", "empty", "load", "unload", "pack", "unpack", "seal", "unseal",
            "hammer", "drill", "sand", "paint", "glue", "staple", "clip", "clamp",
            "mix", "stir", "blend", "knead", "whisk", "flip", "season", "serve",
            "calibrate", "reset", "initialize", "configure", "stabilize", "secure",
        ]
        text_lower = text.lower()
        for verb in action_verbs:
            if verb in text_lower:
                return verb
        return "manipulate"

    def _validate_response_quality(self, action_info: dict) -> bool:
        if not action_info:
            return False
        action_name = action_info.get("action_name", "").strip()
        if len(action_name) < 3:
            return False
        invalid = ["n/a", "unknown", "error", "...", "null", "none"]
        if action_name.lower() in invalid:
            return False
        return True
