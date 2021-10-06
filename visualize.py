import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_output(heatmaps: np.ndarray, offsets: np.ndarray, threshold: float = 0.) -> np.ndarray:
    """
    モデルからの出力を描画できる形にパースする関数.

    Args:
        heatmaps (np.ndarray): モデルから出力されたNx9x9x17のジョイントの存在確率マップ
        offsets (np.ndarray): モデルから出力されたNx9x9x34のオフセットベクトル
        threshold (float): キーポイントとする閾値

    Returns:
        np.ndarray: 各キーポイントの座標と可視化を行うかのフラグ. [[x_i, y_i, is_visualize],]
    """
    num_joints = heatmaps.shape[-1]
    keypoints = np.zeros((num_joints, 3), dtype=np.uint16)

    for i in range(num_joints):
        heatmap = heatmaps[..., i]
        max_prob = np.max(heatmap)

        pos_idx = np.squeeze(np.argwhere(heatmap==max_prob))        # 確率値が最大のインデックスを取得
        # 最大確率が複数あった場合，最初のインデックスを使用
        if pos_idx.shape[0] > 2:
            pos_idx = pos_idx[0]

        remap_pos = np.array((pos_idx / 8) * 257, dtype=np.uint16)  # 本来の画素のインデックスに変換

        # モデルの出力であるオフセットを使用して位置を調整
        keypoints[i, 0] = int(remap_pos[0] + offsets[pos_idx[0], pos_idx[1], i])
        keypoints[i, 1] = int(remap_pos[1] + offsets[pos_idx[0], pos_idx[1], i + num_joints])

        if max_prob > threshold:
            if keypoints[i, 0] < 257 and keypoints[i, 1] < 257:
                keypoints[i, 2] = True  # visualize

    return keypoints


def draw_keypoints(img: np.ndarray, keypoints: np.ndarray, ratio: float = None) -> np.ndarray:
    """
    画像にキーポイントを描画する関数.

    Args:
        img (np.ndarray): 画像
        keypoints (np.ndarray): キーポイントのリスト
        ratio (float, optional): 画像のリサイズ比率. Defaults to None.

    Returns:
        np.ndarray: キーポイントが描画された画像
    """
    for i in range(keypoints.shape[0]):
        if keypoints[i, 2]:
            if isinstance(ratio, tuple):
                cv2.circle(img,
                           (int(round(keypoints[i, 1] * ratio[1])), int(round(keypoints[i, 0] * ratio[0]))),
                           2, (0, 255, 255), round(int(1 * ratio[1])))
                continue
            cv2.circle(img, (keypoints[i, 1], keypoints[i, 0]), 2, (0, 255, 255), -1)
    return img


if __name__ == '__main__':
    dummy_heatmaps = np.zeros((9, 9, 17))
    dummy_heatmaps[..., 0][7][7] = 3

    dummy_offsets = np.ones((9, 9, 34))

    keypoints = parse_output(dummy_heatmaps, dummy_offsets, 0)
    # print(keypoints)

    dummy_img = np.zeros((257, 257, 3), dtype=np.uint8)
    img = draw_keypoints(dummy_img, keypoints)
    plt.imshow(img)
    plt.show()
