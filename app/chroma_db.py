import os
import time

import cv2
import shutil
import torch
import numpy as np
from typing import List
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from collections import Counter
from .yolov8_model import YOLOFaceDetector
from PIL import Image


class FaceDB:
    def __init__(self, base_dir: str = str(os.path.join(os.getcwd(), 'user')), db_path: str = "my_vectordb", threshold: float = 0.2):
        """
        Khởi tạo lớp FaceDB để quản lý cơ sở dữ liệu người dùng và ảnh với Chroma DB.
        Khi khởi tạo, tự động thêm tất cả người dùng trong thư mục `user` vào DB, rồi xóa thư mục `user`.

        Args:
            base_dir (str): Thư mục gốc chứa các thư mục người dùng cần thêm.
            db_path (str): Đường dẫn tới Chroma DB.
            threshold (float): Ngưỡng để xác định khi nào một embedding là trùng khớp.
        """
        self.base_dir = base_dir
        self.threshold = threshold
        self.chroma_client = PersistentClient(path=db_path)
        self.image_loader = ImageLoader()
        self.embedding_function = OpenCLIPEmbeddingFunction(model_name= 'MobileCLIP-B',checkpoint="datacompdr") ## threhold = 0.2
        # self.embedding_function = OpenCLIPEmbeddingFunction(model_name='ViT-g-14',
        #                                                     checkpoint="laion2b_s34b_b88k")  ## threhold = 2
        self.face_detector = YOLOFaceDetector()

        # Tạo hoặc lấy collection trong Chroma DB
        self.db = self.chroma_client.get_or_create_collection(
            name="multimodal_db",
            embedding_function=self.embedding_function,
            data_loader=self.image_loader,
        )

        # Tự động add người dùng từ thư mục 'user' khi khởi tạo
        self.load_and_delete_users()
        print(self.db.count())

    def load_and_delete_users(self):
        """
        Tải tất cả người dùng từ thư mục `user`, lưu thông tin vào DB, và xóa thư mục `user`.
        """
        if not os.path.exists(self.base_dir):
            print(f"Thư mục '{self.base_dir}' không tồn tại.")
            return
        print(self.base_dir)

        # Duyệt qua tất cả các thư mục người dùng trong `base_dir`
        for user_name in os.listdir(self.base_dir):
            user_dir = os.path.join(self.base_dir, user_name)
            print("User_dir name:", user_name)
            if os.path.isdir(user_dir):
                # Tải ảnh người dùng và thêm vào DB
                # images = self.load_images_from_folder(user_dir)
                images_uri = self.load_image_paths_from_folder(folder_path=user_dir)
                self.add_user_to_db(user_name, images_uri)
                print(f"Đã thêm người dùng '{user_name}' vào cơ sở dữ liệu với {len(images_uri)} ảnh.")

        # Xóa thư mục `user` sau khi xử lý xong
        # shutil.rmtree(self.base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"Đã xóa thư mục '{self.base_dir}' sau khi lưu thông tin người dùng.")

    def load_images_from_folder(self, folder_path: str) -> List[np.ndarray]:
        """
        Tải tất cả các ảnh từ một thư mục người dùng dưới dạng numpy arrays.

        Args:
            folder_path (str): Đường dẫn tới thư mục chứa ảnh của người dùng.

        Returns:
            List[np.ndarray]: Danh sách các ảnh dưới dạng numpy arrays.
        """
        print("Load ảnh tại:", folder_path)
        images = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img is not None:
                    images.append(img)
                else:
                    print(f"Không thể tải ảnh '{img_path}'")
        return images

    def load_image_paths_from_folder(self, folder_path: str) -> List[str]:
        """
        Lấy danh sách đường dẫn URI của tất cả các ảnh trong một thư mục.

        Args:
            folder_path (str): Đường dẫn tới thư mục chứa ảnh của người dùng.

        Returns:
            List[str]: Danh sách các đường dẫn ảnh.
        """
        print("Lấy đường dẫn ảnh tại:", folder_path)
        image_paths = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            if os.path.isfile(img_path):
                image_paths.append(img_path)
            else:
                print(f"Không tìm thấy file ảnh '{img_path}'")
        print("Path:", image_paths)
        return image_paths

    def add_user_to_db(self, user_name: str, images: List[str]):
        """
        Thêm người dùng và ảnh vào DB với metadata là tên người dùng và số lượng ảnh.

        Args:
            user_name (str): Tên người dùng để lưu.
            images (List[np.ndarray]): Danh sách các ảnh (numpy array) cần lưu.
        """

        print("Adding add_user_to_db:", user_name)
        # Tạo metadata và id cho mỗi ảnh, rồi thêm vào DB
        for i, image in enumerate(images):

            print(image)
            metadata = {"name": user_name, "image_id": f"image_{i}"}
            image_id = f"{user_name}_image_{i}"
            # load image from uri
            img_np = cv2.imread(image)
            # img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            result = self.face_detector.detect_faces(img=img_np)[0]
            print(result)

            x_min, y_min, x_max, y_max = result["bounding_box"]

            face_img = Image.open(image).crop((x_min, y_min, x_max, y_max))

            # face_img.save(f"{user_name}_{metadata['image_id']}.jpg")

            face_img_np = np.array(face_img)

            print("Meta data: ", metadata)
            self.db.add(
                metadatas=[metadata],
                ids=[image_id],
                images=[face_img_np],
            )

    def track(self, image_np: np.ndarray):
        """
        Kiểm tra xem ảnh đầu vào có trùng khớp với bất kỳ người dùng nào trong DB.

        Args:
            image_np (np.ndarray): Ảnh đầu vào để kiểm tra.

        Returns:
            Tuple[str, Optional[Dict]]: Tên người dùng và metadata nếu trùng khớp, nếu không thì trả về "unknown".
        """

        start_time = time.time()
        query_result = self.db.query(
            query_images=[image_np],
            n_results=10
        )
        print('Query time:', time.time() - start_time)

        # Lấy danh sách tên duy nhất từ metadatas
        unique_names = list(set([i['name'] for i in query_result['metadatas'][0]]))

        # Tạo từ điển rỗng để chứa tên người dùng và danh sách khoảng cách
        matching_users = {}

        # Lặp qua từng khoảng cách và kiểm tra nếu thỏa ngưỡng threshold
        for i, distance in enumerate(query_result['distances'][0]):
            print("Distance:", distance)
            if distance <= self.threshold:
                user_name = query_result['metadatas'][0][i]['name']
                # Đảm bảo rằng user_name có một danh sách trong từ điển trước khi gọi append
                matching_users.setdefault(user_name, []).append(distance)

        # print(matching_users)
        for num, value in matching_users.items():
            try:
                matching_users[num] = sum(matching_users[num]) / len(matching_users[num])
            except:
                matching_users[num] = 9999

        matching_users_sort = {k: v for k, v in sorted(matching_users.items(), key=lambda item: item[1])}

        print(matching_users_sort)

        # Lấy danh sách người dùng và khoảng cách của họ với ảnh đầu vào

        if not matching_users:
            print("Không phát hiện người dùng trong DB. Trả về 'unknown'.")
            return "unknown", None
        else:
            # Tìm người xuất hiện nhiều nhất trong danh sách
            most_common_user,_ = next(iter(matching_users_sort.items()))
            print("Đã phát hiện người dùng phổ biến nhất:", most_common_user)

            # Trả về tên và metadata của người được phát hiện nhiều nhất
            most_common_metadata = next(
                (metadata for metadata in query_result['metadatas'][0]
                 if metadata['name'] == most_common_user),
                None
            )
            return most_common_user, most_common_metadata
