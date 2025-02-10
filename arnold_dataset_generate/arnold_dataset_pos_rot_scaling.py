from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


# 쿼터니언 관련 함수들
def quaternion_multiply(q, r):
    """
    두 쿼터니언 q와 r을 곱합니다.
    Parameters:
        q (array-like): 첫 번째 쿼터니언 [w, x, y, z].
        r (array-like): 두 번째 쿼터니언 [w, x, y, z].
    Returns:
        numpy.ndarray: 곱셈 결과 쿼터니언.
    """
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def quaternion_conjugate(q):
    """
    쿼터니언의 켤레를 계산합니다.
    Parameters:
        q (array-like): 쿼터니언 [w, x, y, z].
    Returns:
        numpy.ndarray: 켤레 쿼터니언 [w, -x, -y, -z].
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quaternion_inverse(q):
    """
    쿼터니언의 역수를 계산합니다.
    Parameters:
        q (array-like): 쿼터니언 [w, x, y, z].
    Returns:
        numpy.ndarray: 역수 쿼터니언.
    """
    conjugate = quaternion_conjugate(q)
    norm_sq = np.dot(q, q)
    if norm_sq == 0:
        raise ZeroDivisionError("제로 노름 쿼터니언의 역수를 계산할 수 없습니다.")
    return conjugate / norm_sq


def quaternion_to_euler(q):
    """
    정규화된 쿼터니언 q = [w, x, y, z]를 오일러 각도 (roll, pitch, yaw)로 변환합니다.
    반환되는 각도는 라디안 단위입니다.
    """
    w, x, y, z = q
    # Roll (x축 회전)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y축 회전)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z축 회전)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw], dtype=np.float32)


class ArnoldDataset(tfds.core.GeneratorBasedBuilder):
    """데이터셋 빌더 for example dataset."""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """데이터셋 메타데이터"""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.'
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.'
                        ),
                        # state: 위치(3) + quaternion(4) + 현재 gripper_open(1) = [8] shape
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Position(3), quaternion(4)와 현재 gripper_open(1)을 이어붙인 값'
                        ),
                    }),
                    # action: pos_diff (3) + euler_angles (3) + gripper (1) = [7]
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Delta state: [pos_diff (3), euler_angles (3), gripper (1)]'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount 값, 기본 1.0'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='최종 step에만 1.0의 reward'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='첫 번째 step 여부'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='마지막 step 여부'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='터미널 step 여부'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='원본 데이터 파일의 경로'
                    ),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """데이터 분할 정의."""
        import os
        import glob

        # 실제 데이터가 있는 base 경로로 수정해주세요.
        base_path = '/home/sm32289/challenge_data_train'
        base_path_val = '/home/sm32289/data_for_challenge_val'
        # 총 8개 task 목록
        tasks = [
            'pickup_object',
            'reorient_object',
            'close_drawer',
            'open_drawer',
            'close_cabinet',
            'open_cabinet',
            'pour_water',
            'transfer_water'
        ]

        train_files = []
        for task in tasks:
            train_dir = os.path.join(base_path, task, 'train')
            train_files.extend(glob.glob(os.path.join(train_dir, '*.npz')))

        val_files = []
        for task in tasks:
            val_dir = os.path.join(base_path_val, task, 'val')
            val_files.extend(glob.glob(os.path.join(val_dir, '*.npz')))

        return {
            'train': self._generate_examples(train_files),
            'val': self._generate_examples(val_files),
        }

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """각 split에 대한 예제를 생성합니다."""
        if isinstance(paths, str):
            episode_paths = glob.glob(paths)
        else:
            episode_paths = paths

        # position scaling을 위한 최소/최대값 (robot_actions의 lower, upper bound)
        pos_min = np.array([-53.90220865, -46.68493673, -56.90943265], dtype=np.float32)
        pos_max = np.array([77.29988551, 106.42815944, 59.01504069], dtype=np.float32)

        for file_path in episode_paths:
            file = np.load(file_path, allow_pickle=True)
            gt_data = file['gt']
            # 언어 지시문은 모든 step에 대해 동일하게 gt_data[0]['instruction'] 사용
            language_instruction = gt_data[0]['instruction']
            language_embedding = self._embed([language_instruction])[0].numpy().astype(np.float32)

            steps = []
            if ('pour_water' in file_path) or ('transfer_water' in file_path):
                # ----- water task 처리 -----
                # 첫 번째 step
                current = gt_data[0]
                next_data = gt_data[2]
                state = np.concatenate([
                    np.asarray(current['position_rotation_world'][0], dtype=np.float32),
                    np.asarray(current['position_rotation_world'][1], dtype=np.float32),
                    np.array([1.0 if current['gripper_open'] else 0.0], dtype=np.float32)
                ])
                image = current['images'][2]['rgb'][:,:,:3]
                wrist_image = current['images'][4]['rgb'][:,:,:3]

                # delta position 계산 후 scaling
                pos_diff = (np.asarray(next_data['position_rotation_world'][0], dtype=np.float32) -
                            np.asarray(current['position_rotation_world'][0], dtype=np.float32))
                pos_diff_scaled = 2 * (pos_diff - pos_min) / (pos_max - pos_min) - 1

                q_current = np.asarray(current['position_rotation_world'][1], dtype=np.float32)
                q_next = np.asarray(next_data['position_rotation_world'][1], dtype=np.float32)
                q_diff = quaternion_multiply(q_next, quaternion_inverse(q_current))
                q_diff_normalized = q_diff / np.linalg.norm(q_diff)
                euler_angles = quaternion_to_euler(q_diff_normalized)
                # roll, pitch, yaw에 대해 각각 scaling (roll, yaw: x2, pitch: x4)
                scaled_euler_angles = np.array([2 * euler_angles[0],
                                                4 * euler_angles[1],
                                                2 * euler_angles[2]], dtype=np.float32)
                joint_velocity = np.concatenate([pos_diff_scaled, scaled_euler_angles]).astype(np.float32)
                gripper_next = 1.0 if next_data['gripper_open'] else 0.0
                action = np.concatenate([joint_velocity, np.array([gripper_next], dtype=np.float32)]).astype(np.float32)

                step0 = {
                    'observation': {
                        'image': image,
                        'wrist_image': wrist_image,
                        'state': state.astype(np.float32),
                    },
                    'action': action,
                    'discount': np.float32(1.0),
                    'reward': np.float32(0.0),  # 첫 번째 step reward 0
                    'is_first': True,
                    'is_last': False,
                    'is_terminal': False,
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                }
                steps.append(step0)

                # 두 번째 step
                current = gt_data[2]
                pos_diff = (np.asarray(gt_data[3]['position_rotation_world'][0], dtype=np.float32) -
                            np.asarray(current['position_rotation_world'][0], dtype=np.float32))
                pos_diff_scaled = 2 * (pos_diff - pos_min) / (pos_max - pos_min) - 1
                q_current = np.asarray(current['position_rotation_world'][1], dtype=np.float32)
                q_next = np.asarray(gt_data[4]['position_rotation_world'][1], dtype=np.float32)
                q_diff = quaternion_multiply(q_next, quaternion_inverse(q_current))
                q_diff_normalized = q_diff / np.linalg.norm(q_diff)
                euler_angles = quaternion_to_euler(q_diff_normalized)
                scaled_euler_angles = np.array([2 * euler_angles[0],
                                                4 * euler_angles[1],
                                                2 * euler_angles[2]], dtype=np.float32)
                joint_velocity = np.concatenate([pos_diff_scaled, scaled_euler_angles]).astype(np.float32)
                gripper_next = 1.0 if gt_data[3]['gripper_open'] else 0.0
                action = np.concatenate([joint_velocity, np.array([gripper_next], dtype=np.float32)]).astype(np.float32)

                state = np.concatenate([
                    np.asarray(current['position_rotation_world'][0], dtype=np.float32),
                    np.asarray(current['position_rotation_world'][1], dtype=np.float32),
                    np.array([1.0 if current['gripper_open'] else 0.0], dtype=np.float32)
                ])
                image = current['images'][2]['rgb'][:,:,:3]
                wrist_image = current['images'][4]['rgb'][:,:,:3]

                step1 = {
                    'observation': {
                        'image': image,
                        'wrist_image': wrist_image,
                        'state': state.astype(np.float32),
                    },
                    'action': action,
                    'discount': np.float32(1.0),
                    'reward': np.float32(1.0),  # 마지막 step reward 1
                    'is_first': False,
                    'is_last': True,
                    'is_terminal': True,
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                }
                steps.append(step1)
            else:
                # ----- 기타 task 처리 -----
                # 첫 번째 step
                current = gt_data[0]
                next_data = gt_data[2]
                state = np.concatenate([
                    np.asarray(current['position_rotation_world'][0], dtype=np.float32),
                    np.asarray(current['position_rotation_world'][1], dtype=np.float32),
                    np.array([1.0 if current['gripper_open'] else 0.0], dtype=np.float32)
                ])
                image = current['images'][2]['rgb'][:,:,:3]
                wrist_image = current['images'][4]['rgb'][:,:,:3]

                pos_diff = (np.asarray(next_data['position_rotation_world'][0], dtype=np.float32) -
                            np.asarray(current['position_rotation_world'][0], dtype=np.float32))
                pos_diff_scaled = 2 * (pos_diff - pos_min) / (pos_max - pos_min) - 1
                q_current = np.asarray(current['position_rotation_world'][1], dtype=np.float32)
                q_next = np.asarray(next_data['position_rotation_world'][1], dtype=np.float32)
                q_diff = quaternion_multiply(q_next, quaternion_inverse(q_current))
                q_diff_normalized = q_diff / np.linalg.norm(q_diff)
                euler_angles = quaternion_to_euler(q_diff_normalized)
                scaled_euler_angles = np.array([2 * euler_angles[0],
                                                4 * euler_angles[1],
                                                2 * euler_angles[2]], dtype=np.float32)
                joint_velocity = np.concatenate([pos_diff_scaled, scaled_euler_angles]).astype(np.float32)
                gripper_next = 1.0 if next_data['gripper_open'] else 0.0
                action = np.concatenate([joint_velocity, np.array([gripper_next], dtype=np.float32)]).astype(np.float32)

                step0 = {
                    'observation': {
                        'image': image,
                        'wrist_image': wrist_image,
                        'state': state.astype(np.float32),
                    },
                    'action': action,
                    'discount': np.float32(1.0),
                    'reward': np.float32(0.0),
                    'is_first': True,
                    'is_last': False,
                    'is_terminal': False,
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                }
                steps.append(step0)

                # 두 번째 step
                current = gt_data[2]
                next_data = gt_data[3]
                state = np.concatenate([
                    np.asarray(current['position_rotation_world'][0], dtype=np.float32),
                    np.asarray(current['position_rotation_world'][1], dtype=np.float32),
                    np.array([1.0 if current['gripper_open'] else 0.0], dtype=np.float32)
                ])
                image = current['images'][2]['rgb'][:,:,:3]
                wrist_image = current['images'][4]['rgb'][:,:,:3]

                pos_diff = (np.asarray(next_data['position_rotation_world'][0], dtype=np.float32) -
                            np.asarray(current['position_rotation_world'][0], dtype=np.float32))
                pos_diff_scaled = 2 * (pos_diff - pos_min) / (pos_max - pos_min) - 1
                q_current = np.asarray(current['position_rotation_world'][1], dtype=np.float32)
                q_next = np.asarray(next_data['position_rotation_world'][1], dtype=np.float32)
                q_diff = quaternion_multiply(q_next, quaternion_inverse(q_current))
                q_diff_normalized = q_diff / np.linalg.norm(q_diff)
                euler_angles = quaternion_to_euler(q_diff_normalized)
                scaled_euler_angles = np.array([2 * euler_angles[0],
                                                4 * euler_angles[1],
                                                2 * euler_angles[2]], dtype=np.float32)
                joint_velocity = np.concatenate([pos_diff_scaled, scaled_euler_angles]).astype(np.float32)
                gripper_next = 1.0 if next_data['gripper_open'] else 0.0
                action = np.concatenate([joint_velocity, np.array([gripper_next], dtype=np.float32)]).astype(np.float32)

                step1 = {
                    'observation': {
                        'image': image,
                        'wrist_image': wrist_image,
                        'state': state.astype(np.float32),
                    },
                    'action': action,
                    'discount': np.float32(1.0),
                    'reward': np.float32(1.0),
                    'is_first': False,
                    'is_last': True,
                    'is_terminal': True,
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                }
                steps.append(step1)
            
            sample = {
                'steps': steps,
                'episode_metadata': {
                    'file_path': file_path
                }
            }
            yield file_path, sample


Builder = ArnoldDataset
