package net.philip.face.mtcnn;

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.Vector;

import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.springframework.core.io.DefaultResourceLoader;
import org.tensorflow.framework.ConfigProto;

public class MTCNN {
	// 参数
	private float factor = 0.709f;
	private float PNetThreshold = 0.6f;
	private float RNetThreshold = 0.7f;
	private float ONetThreshold = 0.7f;
	// MODEL PATH
	private static final String MODEL_FILE = "model/mtcnn_freezed_model.pb";

	private GraphRunner proposeNetGraphRunner = null;
	private GraphRunner refineNetGraphRunner = null;
	private GraphRunner outputNetGraphRunner = null;

	// tensor name
	private static final String PNetInName = "pnet/input:0";
	private static final String[] PNetOutName = new String[] { "pnet/prob1:0", "pnet/conv4-2/BiasAdd:0" };
	private static final String RNetInName = "rnet/input:0";
	private static final String[] RNetOutName = new String[] { "rnet/prob1:0", "rnet/conv5-2/conv5-2:0", };
	private static final String ONetInName = "onet/input:0";
	private static final String[] ONetOutName = new String[] { "onet/prob1:0", "onet/conv6-2/conv6-2:0",
			"onet/conv6-3/conv6-3:0" };
	// 安卓相关
	public long lastProcessTime; // �?后一张图片处理的时间ms

	public void loadModel() throws Exception {

		this.proposeNetGraphRunner = this.createGraphRunner(MODEL_FILE, PNetInName, PNetOutName);
		this.refineNetGraphRunner = this.createGraphRunner(MODEL_FILE, RNetInName, RNetOutName);
		this.outputNetGraphRunner = this.createGraphRunner(MODEL_FILE, ONetInName, ONetOutName);
	}

	private GraphRunner createGraphRunner(String tensorflowModelUri, String inputLabel, String... outLabel) {
		try {
			ConfigProto cp = ConfigProto.newBuilder()
					.setInterOpParallelismThreads(Runtime.getRuntime().availableProcessors() * 2)
					.setAllowSoftPlacement(true).setLogDevicePlacement(true).build();
			return new GraphRunner(
					IOUtils.toByteArray(new DefaultResourceLoader().getResource(tensorflowModelUri).getInputStream()),
					Arrays.asList(inputLabel), Arrays.asList(outLabel), cp);
		} catch (IOException e) {
			throw new IllegalStateException(
					String.format("Failed to load TF model [%s] and input [%s]:", tensorflowModelUri, inputLabel), e);
		}
	}

	// 读取BufferedImage像素值，预处�?(-127.5 /128)，转化为�?维数组返�?
	private float[] normalizeImage(BufferedImage image) {
		int w = image.getWidth();
		int h = image.getHeight();
		float[] floatValues = new float[w * h * 3];
		int[] intValues = new int[w * h];
		// BufferedImage.getPixels(intValues,0,BufferedImage.getWidth(),0,0,BufferedImage.getWidth(),BufferedImage.getHeight());

		// pixel = rgbArray[offset + (y-startY)*scansize + (x-startX)];
		image.getRGB(0, 0, w, h, intValues, 0, w);

		float imageMean = 127.5f;
		float imageStd = 128;

		for (int i = 0; i < intValues.length; i++) {
			final int val = intValues[i];
			floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
			floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
			floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
		}
		return floatValues;
	}

	/*
	 * �?测人�?,minSize是最小的人脸像素�?
	 */
	private BufferedImage bufferedImageResize(BufferedImage bm, float scale) {

		int newWidth = (int) (bm.getWidth() * scale);
		int newHeight = (int) (bm.getHeight() * scale);
		BufferedImage resizeImg = new BufferedImage(newWidth, newHeight, bm.getType());
		Graphics2D g = resizeImg.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		g.drawImage(bm, 0, 0, newWidth, newHeight, 0, 0, bm.getWidth(), bm.getHeight(), null);
		g.dispose();

		return resizeImg;
	}

	private BufferedImage bufferedImageResize(BufferedImage bm, float scale, int left, int top, int width, int height) {

		int newWidth = (int) (width * scale);
		int newHeight = (int) (height * scale);
		BufferedImage resizeImg = new BufferedImage(newWidth, newHeight, bm.getType());
		Graphics2D g = resizeImg.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		g.drawImage(bm, 0, 0, newWidth, newHeight, left, top, left+width, top+height, null);
		g.dispose();

		return resizeImg;
	}

	// 输入前要翻转，输出也要翻�?
	private int PNetForward(BufferedImage image, float[][] PNetOutProb, float[][][] PNetOutBias) {
		int w = image.getWidth();
		int h = image.getHeight();

		float[] PNetIn = normalizeImage(image);
		//翻转后变成w*h*stride
		Utils.flip_diag(PNetIn, h, w, 3); // 沿着对角线翻�?
		// inferenceInterface.feed(PNetInName,PNetIn,1,w,h,3);
		// inferenceInterface.run(PNetOutName,false);

		INDArray pIn = Nd4j.create(PNetIn, new int[] {1, w, h, 3 });
		
		Map<String, INDArray> resultMap = this.proposeNetGraphRunner.run(Collections.singletonMap(PNetInName, pIn));
		INDArray out0 = resultMap.get(PNetOutName[0]);
		INDArray out1 = resultMap.get(PNetOutName[1]);

		int PNetOutSizeW = (int) Math.ceil(w * 0.5 - 5);
		int PNetOutSizeH = (int) Math.ceil(h * 0.5 - 5);
		float[] PNetOutP = new float[PNetOutSizeW * PNetOutSizeH * 2];
		float[] PNetOutB = new float[PNetOutSizeW * PNetOutSizeH * 4];
		// inferenceInterface.fetch(PNetOutName[0],PNetOutP);
		// inferenceInterface.fetch(PNetOutName[1],PNetOutB);

		
//		Nd4j.toFlattened(out0).
		PNetOutP = out0.data().asFloat();
		PNetOutB = out1.data().asFloat();

		// 【写法一】先翻转，后转为2/3维数�?
		Utils.flip_diag(PNetOutP, PNetOutSizeW, PNetOutSizeH, 2);
		Utils.flip_diag(PNetOutB, PNetOutSizeW, PNetOutSizeH, 4);
		Utils.expand(PNetOutB, PNetOutBias);
		Utils.expandProb(PNetOutP, PNetOutProb);
		/*
		 * 【写法二】这个比较快，快�?3ms。意义不大，用上面的方法比较直观 for (int y=0;y<PNetOutSizeH;y++) for
		 * (int x=0;x<PNetOutSizeW;x++){ int idx=PNetOutSizeH*x+y;
		 * PNetOutProb[y][x]=PNetOutP[idx*2+1]; for(int i=0;i<4;i++)
		 * PNetOutBias[y][x][i]=PNetOutB[idx*4+i]; }
		 */
		return 0;
	}

	// Non-Maximum Suppression
	// nms，不符合条件的deleted设置为true
	private void nms(Vector<Box> boxes, float threshold, String method) {
		// NMS.两两比对
		// int delete_cnt=0;
		int cnt = 0;
		for (int i = 0; i < boxes.size(); i++) {
			Box box = boxes.get(i);
			if (!box.deleted) {
				// score<0表示当前矩形框被删除
				for (int j = i + 1; j < boxes.size(); j++) {
					Box box2 = boxes.get(j);
					if (!box2.deleted) {
						int x1 = max(box.box[0], box2.box[0]);
						int y1 = max(box.box[1], box2.box[1]);
						int x2 = min(box.box[2], box2.box[2]);
						int y2 = min(box.box[3], box2.box[3]);
						if (x2 < x1 || y2 < y1)
							continue;
						int areaIoU = (x2 - x1 + 1) * (y2 - y1 + 1);
						float iou = 0f;
						if (method.equals("Union"))
							iou = 1.0f * areaIoU / (box.area() + box2.area() - areaIoU);
						else if (method.equals("Min")) {
							iou = 1.0f * areaIoU / (min(box.area(), box2.area()));
						}
						if (iou >= threshold) { // 删除prob小的那个�?
							if (box.score > box2.score)
								box2.deleted = true;
							else
								box.deleted = true;
							// delete_cnt++;
						}
					}
				}
			}
		}
		// Log.i(TAG,"[*]sum:"+boxes.size()+" delete:"+delete_cnt);
	}

	private int generateBoxes(float[][] prob, float[][][] bias, float scale, float threshold, Vector<Box> boxes) {
		int h = prob.length;
		int w = prob[0].length;
		// Log.i(TAG,"[*]height:"+prob.length+" width:"+prob[0].length);
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++) {
				float score = prob[y][x];
				// only accept prob >threadshold(0.6 here)
				if (score > threshold) {
					Box box = new Box();
					// score
					box.score = score;
					// box
					box.box[0] = Math.round(x * 2 / scale);
					box.box[1] = Math.round(y * 2 / scale);
					box.box[2] = Math.round((x * 2 + 11) / scale);
					box.box[3] = Math.round((y * 2 + 11) / scale);
					// bbr
					for (int i = 0; i < 4; i++)
						box.bbr[i] = bias[y][x][i];
					// add
					boxes.addElement(box);
				}
			}
		return 0;
	}

	private void BoundingBoxReggression(Vector<Box> boxes) {
		for (int i = 0; i < boxes.size(); i++)
			boxes.get(i).calibrate();
	}
	
	// Pnet + Bounding Box Regression + Non-Maximum Regression
	/*
	 * NMS执行完后，才执行Regression (1) For each scale , use NMS with threshold=0.5 (2)
	 * For all candidates , use NMS with threshold=0.7 (3) Calibrate Bounding
	 * Box 注意：CNN输入图片�?上面�?行，坐标为[0..width,0]。所以BufferedImage�?要对折后再跑网络;网络输出同理.
	 */
	private Vector<Box> PNet(BufferedImage image, int minSize) {
		int whMin = min(image.getWidth(), image.getHeight());
		float currentFaceSize = minSize; // currentFaceSize=minSize/(factor^k)
											// k=0,1,2... until excced whMin
		Vector<Box> totalBoxes = new Vector<Box>();
		// �?1】Image Paramid and Feed to Pnet
		while (currentFaceSize <= whMin) {
			float scale = 12.0f / currentFaceSize;
			// (1)Image Resize
			BufferedImage bm = bufferedImageResize(image, scale);

			int w = bm.getWidth();
			int h = bm.getHeight();
			// (2)RUN CNN
			int PNetOutSizeW = (int) (Math.ceil(w * 0.5 - 5) + 0.5);
			int PNetOutSizeH = (int) (Math.ceil(h * 0.5 - 5) + 0.5);
			float[][] PNetOutProb = new float[PNetOutSizeH][PNetOutSizeW];
			float[][][] PNetOutBias = new float[PNetOutSizeH][PNetOutSizeW][4];
			PNetForward(bm, PNetOutProb, PNetOutBias);
			// (3)数据解析
			Vector<Box> curBoxes = new Vector<Box>();
			generateBoxes(PNetOutProb, PNetOutBias, scale, PNetThreshold, curBoxes);
			// Log.i(TAG,"[*]CNN Output Box number:"+curBoxes.size()+"
			// Scale:"+scale);
			// (4)nms 0.5
			nms(curBoxes, 0.5f, "Union");
			// (5)add to totalBoxes
			for (int i = 0; i < curBoxes.size(); i++)
				if (!curBoxes.get(i).deleted)
					totalBoxes.addElement(curBoxes.get(i));
			// Face Size等比递增
			currentFaceSize /= factor;
		}
		// NMS 0.7
		nms(totalBoxes, 0.7f, "Union");
		// BBR
		BoundingBoxReggression(totalBoxes);
		return Utils.updateBoxes(totalBoxes);
	}

	// 截取box中指定的矩形�?(越界要处�?)，并resize到size*size大小，返回数据存放到data中�??
	public BufferedImage tmp_bm;

	private BufferedImage crop_and_resize(BufferedImage BufferedImage, Box box, int size, float[] data) {
		// (2)crop and resize
		// Matrix matrix = new Matrix();
		float scale = 1.0f * size / box.width();
		// matrix.postScale(scale, scale);
		// BufferedImage croped=BufferedImage.createBufferedImage(BufferedImage,
		// box.left(),box.top(),box.width(), box.height(),matrix,true);
		BufferedImage croped = bufferedImageResize(BufferedImage, scale, box.left(), box.top(), box.width(),
				box.height());

		// (3)save
		int[] pixels_buf = new int[size * size];
		// croped.getPixels(pixels_buf,0,croped.getWidth(),0,0,croped.getWidth(),croped.getHeight());
		croped.getRGB(0, 0, croped.getWidth(), croped.getHeight(), pixels_buf, 0, croped.getWidth());
		float imageMean = 127.5f;
		float imageStd = 128;
		for (int i = 0; i < pixels_buf.length; i++) {
			final int val = pixels_buf[i];
			data[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
			data[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
			data[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
		}
		
		return croped;
	}

	/*
	 * RNET跑神经网络，将score和bias写入boxes
	 */
	private void RNetForward(float[] RNetIn, Vector<Box> boxes) {
		int num = RNetIn.length / 24 / 24 / 3;
		// feed & run
		// inferenceInterface.feed(RNetInName,RNetIn,num,24,24,3);
		// inferenceInterface.run(RNetOutName,false);

		INDArray rIn = Nd4j.create(RNetIn, new int[] { num, 24, 24, 3 });
		Map<String, INDArray> resultMap = this.refineNetGraphRunner.run(Collections.singletonMap(RNetInName, rIn));
		INDArray out0 = resultMap.get(RNetOutName[0]);
		INDArray out1 = resultMap.get(RNetOutName[1]);

		// fetch
		float[] RNetP = new float[num * 2];
		float[] RNetB = new float[num * 4];
		// inferenceInterface.fetch(RNetOutName[0],RNetP);
		// inferenceInterface.fetch(RNetOutName[1],RNetB);

		RNetP = out0.data().asFloat();
		RNetB = out1.data().asFloat();

		// 转换
		for (int i = 0; i < num; i++) {
			boxes.get(i).score = RNetP[i * 2 + 1];
			for (int j = 0; j < 4; j++)
				boxes.get(i).bbr[j] = RNetB[i * 4 + j];
		}
	}

	// Refine Net
	private Vector<Box> RNet(BufferedImage BufferedImage, Vector<Box> boxes) {
		// RNet Input Init
		int num = boxes.size();
		float[] RNetIn = new float[num * 24 * 24 * 3];
		float[] curCrop = new float[24 * 24 * 3];
		int RNetInIdx = 0;
		for (int i = 0; i < num; i++) {
			crop_and_resize(BufferedImage, boxes.get(i), 24, curCrop);
			Utils.flip_diag(curCrop, 24, 24, 3);
			// Log.i(TAG,"[*]Pixels values:"+curCrop[0]+" "+curCrop[1]);
			for (int j = 0; j < curCrop.length; j++)
				RNetIn[RNetInIdx++] = curCrop[j];
		}
		// Run RNet
		RNetForward(RNetIn, boxes);
		// RNetThreshold
		for (int i = 0; i < num; i++)
			if (boxes.get(i).score < RNetThreshold)
				boxes.get(i).deleted = true;
		// Nms
		nms(boxes, 0.7f, "Union");
		BoundingBoxReggression(boxes);
		return Utils.updateBoxes(boxes);
	}

	/*
	 * ONet跑神经网络，将score和bias写入boxes
	 */
	private void ONetForward(float[] ONetIn, Vector<Box> boxes) {
		int num = ONetIn.length / 48 / 48 / 3;
		// feed & run
		// inferenceInterface.feed(ONetInName,ONetIn,num,48,48,3);
		// inferenceInterface.run(ONetOutName,false);
		
		INDArray oIn = Nd4j.create(ONetIn, new int[] { num, 48, 48, 3 });
		Map<String, INDArray> resultMap = this.outputNetGraphRunner.run(Collections.singletonMap(ONetInName, oIn));
		INDArray out0 = resultMap.get(ONetOutName[0]);
		INDArray out1 = resultMap.get(ONetOutName[1]);
		INDArray out2 = resultMap.get(ONetOutName[2]);

		// fetch
		float[] ONetP = new float[num * 2]; // prob
		float[] ONetB = new float[num * 4]; // bias
		float[] ONetL = new float[num * 10]; // landmark
		// inferenceInterface.fetch(ONetOutName[0],ONetP);
		// inferenceInterface.fetch(ONetOutName[1],ONetB);
		// inferenceInterface.fetch(ONetOutName[2],ONetL);

		ONetP = out0.data().asFloat();
		ONetB = out1.data().asFloat();
		ONetL = out2.data().asFloat();

		// 转换
		for (int i = 0; i < num; i++) {
			// prob
			boxes.get(i).score = ONetP[i * 2 + 1];
			// bias
			for (int j = 0; j < 4; j++)
				boxes.get(i).bbr[j] = ONetB[i * 4 + j];

			// landmark
			for (int j = 0; j < 5; j++) {
				int x = boxes.get(i).left() + (int) (ONetL[i * 10 + j] * boxes.get(i).width());
				int y = boxes.get(i).top() + (int) (ONetL[i * 10 + j + 5] * boxes.get(i).height());
				boxes.get(i).landmark[j] = new Point(x, y);
				// Log.i(TAG,"[*] landmarkd "+x+ " "+y);
			}
		}
	}

	// ONet
	private Vector<Box> ONet(BufferedImage BufferedImage, Vector<Box> boxes) {
		// ONet Input Init
		int num = boxes.size();
		float[] ONetIn = new float[num * 48 * 48 * 3];
		float[] curCrop = new float[48 * 48 * 3];
		int ONetInIdx = 0;
		for (int i = 0; i < num; i++) {
			crop_and_resize(BufferedImage, boxes.get(i), 48, curCrop);
			Utils.flip_diag(curCrop, 48, 48, 3);
			for (int j = 0; j < curCrop.length; j++)
				ONetIn[ONetInIdx++] = curCrop[j];
		}
		// Run ONet
		ONetForward(ONetIn, boxes);
		// ONetThreshold
		for (int i = 0; i < num; i++)
			if (boxes.get(i).score < ONetThreshold)
				boxes.get(i).deleted = true;
		BoundingBoxReggression(boxes);
		// Nms
		nms(boxes, 0.7f, "Min");
		return Utils.updateBoxes(boxes);
	}

	private void square_limit(Vector<Box> boxes, int w, int h) {
		// square
		for (int i = 0; i < boxes.size(); i++) {
			boxes.get(i).toSquareShape();
			boxes.get(i).limit_square(w, h);
		}
	}

	/*
	 * 参数�? BufferedImage:要处理的图片 minFaceSize:�?小的人脸像素�?.(此�?�越大，�?测越�?) 返回�? 人脸�?
	 */
	public Vector<Box> detectFaces(BufferedImage image, int minFaceSize) {
		long t_start = System.currentTimeMillis();
		// �?1】PNet generate candidate boxes
		Vector<Box> boxes = PNet(image, minFaceSize);
		square_limit(boxes, image.getWidth(), image.getHeight());
		// �?2】RNet
		boxes = RNet(image, boxes);
		square_limit(boxes, image.getWidth(), image.getHeight());
		// �?3】ONet
		boxes = ONet(image, boxes);
		// return
		System.out.println("[*]Mtcnn Detection Time:" + (System.currentTimeMillis() - t_start));
		lastProcessTime = (System.currentTimeMillis() - t_start);
		return boxes;
	}
}
