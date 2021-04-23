import onnxruntime as rt
import numpy as np
import cv2
from PIL import Image
import argparse
import pyclipper
from shapely.geometry import Polygon
from keys import alphabetChinese as alphabet

class SegDetectorRepresenter:
    def __init__(self, thresh=0.3, box_thresh=0.5, max_candidates=1000, unclip_ratio=2.0):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
    def __call__(self, pred, height, width):
        pred = pred[0, :, :]
        segmentation = self.binarize(pred)
        boxes, scores = self.boxes_from_bitmap(pred, segmentation, width, height)
        return boxes, scores

    def binarize(self, pred):
        return pred > self.thresh
    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        assert len(bitmap.shape) == 2
        # bitmap = _bitmap.cpu().numpy()  # The first channel
        # pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        # print(bitmap)
        # cv2.imwrite("./test/output/test.jpg",(bitmap * 255).astype(np.uint8))
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)
        rects = []
        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue
            # print(points)
            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            # print(box)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / (poly.length)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

class DBNET():
    def __init__(self):
        self.sess = rt.InferenceSession('models/dbnet.onnx')
        self.decode_handel = SegDetectorRepresenter()
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
    def process(self, img, short_size=960):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if h < w:
            scale_h = short_size / h
            tar_w = w * scale_h
            tar_w = tar_w - tar_w % 32
            tar_w = max(32, tar_w)
            scale_w = tar_w / w
        else:
            scale_w = short_size / w
            tar_h = h * scale_w
            tar_h = tar_h - tar_h % 32
            tar_h = max(32, tar_h)
            scale_h = tar_h / h

        img = cv2.resize(img, None, fx=scale_w, fy=scale_h)
        img = img.astype(np.float32)

        img /= 255.0
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(img, axis=0)
        out = self.sess.run(["out1"], {"input0": transformed_image.astype(np.float32)})
        box_list, score_list = self.decode_handel(out[0][0], h, w)
        if len(box_list) > 0:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
            box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []
        return box_list, score_list

class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + 'ç'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
    def decode(self, t, length, raw=False):
        t = t[:length]
        if raw:
            return ''.join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return ''.join(char_list)

class CRNNHandle:
    def __init__(self):
        self.sess = rt.InferenceSession('models/crnn_lite_lstm.onnx')
        self.converter = strLabelConverter(''.join(alphabet))
    def predict_rbg(self, im):
        scale = im.size[1] * 1.0 / 32
        w = im.size[0] / scale
        w = int(w)
        img = im.resize((w, 32), Image.BILINEAR)
        img = np.array(img, dtype=np.float32)
        img -= 127.5
        img /= 127.5
        image = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)
        preds = self.sess.run(["out"], {"input": transformed_image.astype(np.float32)})
        preds = preds[0]
        length  = preds.shape[0]
        preds = preds.reshape(length,-1)
        # preds = softmax(preds)
        preds = np.argmax(preds,axis=1)
        preds = preds.reshape(-1)
        sim_pred = self.converter.decode(preds, length, raw=False)
        return sim_pred

class faster_rcnn():
    def __init__(self, confidence=0.5, threshold=0.3, use_nms=False):
        self.confidence = confidence
        self.threshold = threshold
        self.use_nms = use_nms
        self.net = cv2.dnn.readNet('models/faster_rcnn.pb', 'models/faster_rcnn.pbtxt')
    def filter_boxes(self, cvOut, rows, cols):
        boxes = []
        confidences = []
        for detection in cvOut[0, 0, :, :]:
            score = float(detection[2])
            if score > self.confidence:
                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)
                boxes.append((left, top, right - left + 1, bottom - top + 1))
                confidences.append(score)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        return idxs, boxes, confidences
    def detect(self, frame):
        rows = frame.shape[0]
        cols = frame.shape[1]
        blob = cv2.dnn.blobFromImage(frame)
        self.net.setInput(blob)
        cvOut = self.net.forward()
        if self.use_nms:
            idxs, boxes, confidences = self.filter_boxes(cvOut, rows, cols)
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    # draw a bounding box rectangle and label on the image
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
                    # cv2.putText(frame, str(round(confidences[i],3)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            for detection in cvOut[0, 0, :, :]:
                score = float(detection[2])
                if score > self.confidence:
                    left = detection[3] * cols
                    top = detection[4] * rows
                    right = detection[5] * cols
                    bottom = detection[6] * rows
                    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
        return frame

class OCR:
    def __init__(self):
        self.detect = DBNET()
        self.recognition = CRNNHandle()
    def get_rotate_crop_image(self, img, points):
        points += 0.3
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom + 1, left:right + 1, :]
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        img_crop_width = int(np.linalg.norm(points[0] - points[1])) + 1
        img_crop_height = int(np.linalg.norm(points[0] - points[3])) + 1
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height], [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        # dst_img = cv2.warpPerspective(img_crop, M, (img_crop_width, img_crop_height))
        dst_img = cv2.warpPerspective(img_crop, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE)
        return dst_img
    def det_rec(self, srcimg):
        box_list, score_list = self.detect.process(srcimg)
        if len(box_list) == 0:
            return []
        box_list = np.flipud(box_list)
        results = []
        for point in box_list:
            textimg = self.get_rotate_crop_image(srcimg, point.astype(np.float32))
            text = self.recognition.predict_rbg(Image.fromarray(cv2.cvtColor(textimg,cv2.COLOR_BGR2RGB)))
            results.append({'location':point, 'text':text})
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='detect card ocr')
    parser.add_argument('--imgpath', default='demo.png', type=str, help='image path')
    args = parser.parse_args()
    srcimg = cv2.imread(args.imgpath)

    myocr = OCR()
    results = myocr.det_rec(srcimg)
    for i, res in enumerate(results):
        point = res['location'].astype(int)
        cv2.polylines(srcimg, [point], True, (0, 0, 255), thickness=2)
        # for j in range(4):
        #     cv2.circle(srcimg, tuple(point[j, :]), 3, (0, 255, 0), thickness=-1)
        print(res['text'])

    # det_card = faster_rcnn(use_nms=True)
    # srcimg = det_card.detect(srcimg)

    cv2.namedWindow('OCR', cv2.WINDOW_NORMAL)
    cv2.imshow('OCR', srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()