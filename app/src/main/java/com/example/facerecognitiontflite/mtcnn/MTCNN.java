package com.example.facerecognitiontflite.mtcnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Point;

import com.example.facerecognitiontflite.MyUtil;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

/**
 * MTCNN : Face Detection
 */
public class MTCNN {
    private float factor = 0.709f;
    private float pNetThreshold = 0.6f;
    private float rNetThreshold = 0.7f;
    private float oNetThreshold = 0.7f;

    private static final String MODEL_FILE_PNET = "pnet.tflite";
    private static final String MODEL_FILE_RNET = "rnet.tflite";
    private static final String MODEL_FILE_ONET = "onet.tflite";

    private Interpreter pInterpreter;
    private Interpreter rInterpreter;
    private Interpreter oInterpreter;

    public MTCNN(AssetManager assetManager) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        pInterpreter = new Interpreter(MyUtil.loadModelFile(assetManager, MODEL_FILE_PNET), options);
        rInterpreter = new Interpreter(MyUtil.loadModelFile(assetManager, MODEL_FILE_RNET), options);
        oInterpreter = new Interpreter(MyUtil.loadModelFile(assetManager, MODEL_FILE_ONET), options);
    }

    /**
     * Set MTCNN Threshold
     * @param threshold
     */
    public void setThreshold(float threshold){
        this.pNetThreshold = threshold;
        this.rNetThreshold = threshold + 0.1f;
        this.oNetThreshold = threshold + 0.1f;
    }

    /**
     * Face Detection
     * @param bitmap : input bitmap image
     * @param minFaceSize : The smallest face pixel value. (The larger the value, the faster the detection)
     */
    public Vector<Box> detectFaces(Bitmap bitmap, int minFaceSize) {
        Vector<Box> boxes;
        try {
            // [1] pNet generate candidate boxes
            boxes = pNet(bitmap, minFaceSize);
            square_limit(boxes, bitmap.getWidth(), bitmap.getHeight());

            // [2] rNet
            boxes = rNet(bitmap, boxes);
            square_limit(boxes, bitmap.getWidth(), bitmap.getHeight());

            // [3] oNet
            boxes = oNet(bitmap, boxes);
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
            boxes = new Vector<>();
        }
        return boxes;
    }

    private void square_limit(Vector<Box> boxes, int w, int h) {
        // square
        for (int i = 0; i < boxes.size(); i++) {
            boxes.get(i).toSquareShape();
            boxes.get(i).limitSquare(w, h);
        }
    }

    /**
     * Regression is executed after NMS is executed
     * (1) For each scale , use NMS with threshold=0.5
     * (2) For all candidates , use NMS with threshold=0.7
     * (3) Calibrate Bounding Box
     * Note: The top line of the CNN input picture, the coordinate is [0..width, 0].
     * Therefore, Bitmap needs to be folded in half before running the network; the network output is the same.
     *
     * @param bitmap : input bitmap image
     * @return
     */
    private Vector<Box> pNet(Bitmap bitmap, int minSize) {
        int whMin = Math.min(bitmap.getWidth(), bitmap.getHeight());
        float currentFaceSize = minSize; // currentFaceSize=minSize/(factor^k) k=0,1,2... until excced whMin
        Vector<Box> totalBoxes = new Vector<>();
        // [1] Image Paramid and Feed to pNet
        while (currentFaceSize <= whMin) {
            float scale = 12.0f / currentFaceSize;

            // (1) Image Resize
            Bitmap bm = MyUtil.bitmapResize(bitmap, scale);
            int w = bm.getWidth();
            int h = bm.getHeight();

            // (2) RUN CNN
            int outW = (int) (Math.ceil(w * 0.5 - 5) + 0.5);
            int outH = (int) (Math.ceil(h * 0.5 - 5) + 0.5);
            float[][][][] prob1 = new float[1][outW][outH][2];
            float[][][][] conv4_2_BiasAdd = new float[1][outW][outH][4];
            pNetForward(bm, prob1, conv4_2_BiasAdd);
            prob1 = MyUtil.transposeBatch(prob1);
            conv4_2_BiasAdd = MyUtil.transposeBatch(conv4_2_BiasAdd);

            // (3) Generate Boxes
            Vector<Box> curBoxes = new Vector<>();
            generateBoxes(prob1, conv4_2_BiasAdd, scale, curBoxes);

            // (4) NMS with 0.5 of threshold
            nms(curBoxes, 0.5f, "Union");

            // (5) Add to totalBoxes
            for (int i = 0; i < curBoxes.size(); i++)
                if (!curBoxes.get(i).deleted)
                    totalBoxes.addElement(curBoxes.get(i));

            // Face Size increases proportionally
            currentFaceSize /= factor;
        }

        // NMS 0.7
        nms(totalBoxes, 0.7f, "Union");

        // Bounding Box Regression
        BoundingBoxReggression(totalBoxes);

        return updateBoxes(totalBoxes);
    }

    /**
     * pNet forward propagation
     *
     * @param bitmap : input bitmap image
     * @param prob1 : output probability
     * @param conv4_2_BiasAdd
     * @return
     */
    private void pNetForward(Bitmap bitmap, float[][][][] prob1, float[][][][] conv4_2_BiasAdd) {
        float[][][] img = MyUtil.normalizeImage(bitmap);
        float[][][][] pNetIn = new float[1][][][];
        pNetIn[0] = img;
        pNetIn = MyUtil.transposeBatch(pNetIn);

        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(pInterpreter.getOutputIndex("pnet/prob1"), prob1);
        outputs.put(pInterpreter.getOutputIndex("pnet/conv4-2/BiasAdd"), conv4_2_BiasAdd);

        pInterpreter.runForMultipleInputsOutputs(new Object[]{pNetIn}, outputs);
    }

    private void generateBoxes(float[][][][] prob1, float[][][][] conv4_2_BiasAdd, float scale, Vector<Box> boxes) {
        int h = prob1[0].length;
        int w = prob1[0][0].length;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float score = prob1[0][y][x][1];
                // Only accept prob > threshold
                if (score > pNetThreshold) {
                    Box box = new Box();
                    // Score
                    box.score = score;
                    // Bounding box
                    box.box[0] = Math.round(x * 2 / scale);
                    box.box[1] = Math.round(y * 2 / scale);
                    box.box[2] = Math.round((x * 2 + 11) / scale);
                    box.box[3] = Math.round((y * 2 + 11) / scale);
                    // Bounding Box Regression
                    for (int i = 0; i < 4; i++) {
                        box.bbr[i] = conv4_2_BiasAdd[0][y][x][i];
                    }
                    // Add bounding box
                    boxes.addElement(box);
                }
            }
        }
    }

    /**
     * Non-Max Suppression
     *
     * @param boxes : Bounding boxes
     * @param threshold : Threshold for NMS
     * @param method : NMS Method
     */
    private void nms(Vector<Box> boxes, float threshold, String method) {
        // int delete_cnt = 0;
        for (int i = 0; i < boxes.size(); i++) {
            Box box = boxes.get(i);
            if (!box.deleted) {
                // score < 0 : Indicates that the current rectangular frame is deleted
                for (int j = i + 1; j < boxes.size(); j++) {
                    Box box2 = boxes.get(j);
                    if (!box2.deleted) {
                        int x1 = Math.max(box.box[0], box2.box[0]);
                        int y1 = Math.max(box.box[1], box2.box[1]);
                        int x2 = Math.min(box.box[2], box2.box[2]);
                        int y2 = Math.min(box.box[3], box2.box[3]);
                        if (x2 < x1 || y2 < y1) continue;
                        int areaIoU = (x2 - x1 + 1) * (y2 - y1 + 1);
                        float iou = 0f;
                        if (method.equals("Union"))
                            iou = 1.0f * areaIoU / (box.area() + box2.area() - areaIoU);
                        else if (method.equals("Min"))
                            iou = 1.0f * areaIoU / (Math.min(box.area(), box2.area()));
                        if (iou >= threshold) { // Delete the box with the smaller prob
                            if (box.score > box2.score)
                                box2.deleted = true;
                            else
                                box.deleted = true;
                        }
                    }
                }
            }
        }
    }

    private void BoundingBoxReggression(Vector<Box> boxes) {
        for (int i = 0; i < boxes.size(); i++)
            boxes.get(i).calibrate();
    }

    /**
     * Refine Net
     * @param bitmap : input bitmap image
     * @param boxes : Bounding boxes
     * @return
     */
    private Vector<Box> rNet(Bitmap bitmap, Vector<Box> boxes) {
        // rNet Input Init
        int num = boxes.size();
        float[][][][] rNetIn = new float[num][24][24][3];
        for (int i = 0; i < num; i++) {
            float[][][] curCrop = MyUtil.cropAndResize(bitmap, boxes.get(i), 24);
            curCrop = MyUtil.transposeImage(curCrop);
            rNetIn[i] = curCrop;
        }

        // Run rNet
        rNetForward(rNetIn, boxes);

        // rNet Threshold
        for (int i = 0; i < num; i++) {
            if (boxes.get(i).score < rNetThreshold) {
                boxes.get(i).deleted = true;
            }
        }

        // Nms
        nms(boxes, 0.7f, "Union");
        BoundingBoxReggression(boxes);
        return updateBoxes(boxes);
    }

    /**
     * Runs rNET then write score and bias into boxes
     * @param rNetIn
     * @param boxes
     */
    private void rNetForward(float[][][][] rNetIn, Vector<Box> boxes) {
        int num = rNetIn.length;
        float[][] prob1 = new float[num][2];
        float[][] conv5_2_conv5_2 = new float[num][4];

        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(rInterpreter.getOutputIndex("rnet/prob1"), prob1);
        outputs.put(rInterpreter.getOutputIndex("rnet/conv5-2/conv5-2"), conv5_2_conv5_2);
        rInterpreter.runForMultipleInputsOutputs(new Object[]{rNetIn}, outputs);

        // Conversion
        for (int i = 0; i < num; i++) {
            boxes.get(i).score = prob1[i][1];
            for (int j = 0; j < 4; j++) {
                boxes.get(i).bbr[j] = conv5_2_conv5_2[i][j];
            }
        }
    }

    /**
     * oNet
     * @param bitmap
     * @param boxes
     * @return
     */
    private Vector<Box> oNet(Bitmap bitmap, Vector<Box> boxes) {
        // oNet Input Init
        int num = boxes.size();
        float[][][][] oNetIn = new float[num][48][48][3];
        for (int i = 0; i < num; i++) {
            float[][][] curCrop = MyUtil.cropAndResize(bitmap, boxes.get(i), 48);
            curCrop = MyUtil.transposeImage(curCrop);
            oNetIn[i] = curCrop;
        }

        // Run oNet
        oNetForward(oNetIn, boxes);
        // oNet Threshold
        for (int i = 0; i < num; i++) {
            if (boxes.get(i).score < oNetThreshold) {
                boxes.get(i).deleted = true;
            }
        }
        BoundingBoxReggression(boxes);
        // NMS
        nms(boxes, 0.7f, "Min");
        return updateBoxes(boxes);
    }

    /**
     * Runs oNet then write score and bias into boxes
     * @param oNetIn
     * @param boxes
     */
    private void oNetForward(float[][][][] oNetIn, Vector<Box> boxes) {
        int num = oNetIn.length;
        float[][] prob1 = new float[num][2];
        float[][] conv6_2_conv6_2 = new float[num][4];
        float[][] conv6_3_conv6_3 = new float[num][10];

        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(oInterpreter.getOutputIndex("onet/prob1"), prob1);
        outputs.put(oInterpreter.getOutputIndex("onet/conv6-2/conv6-2"), conv6_2_conv6_2);
        outputs.put(oInterpreter.getOutputIndex("onet/conv6-3/conv6-3"), conv6_3_conv6_3);
        oInterpreter.runForMultipleInputsOutputs(new Object[]{oNetIn}, outputs);

        // Conversion
        for (int i = 0; i < num; i++) {
            // Probability
            boxes.get(i).score = prob1[i][1];
            // Bias
            for (int j = 0; j < 4; j++) {
                boxes.get(i).bbr[j] = conv6_2_conv6_2[i][j];
            }
            // Landmark
            for (int j = 0; j < 5; j++) {
                int x = Math.round(boxes.get(i).left() + (conv6_3_conv6_3[i][j] * boxes.get(i).width()));
                int y = Math.round(boxes.get(i).top() + (conv6_3_conv6_3[i][j + 5] * boxes.get(i).height()));
                boxes.get(i).landmark[j] = new Point(x, y);
            }
        }
    }

    /**
     * Delete the box marked with delete
     * @param boxes : Bounding boxes
     * @return
     */
    public static Vector<Box> updateBoxes(Vector<Box> boxes) {
        Vector<Box> b = new Vector<>();
        for (int i = 0; i < boxes.size(); i++) {
            if (!boxes.get(i).deleted) {
                b.addElement(boxes.get(i));
            }
        }
        return b;
    }
}
