package net.philip.face.facenet;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Vector;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.CollectionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.philip.face.facenet.model.InceptionResNetV1;
import net.philip.face.mtcnn.Box;
import net.philip.face.mtcnn.MTCNN;

public class Recognizer {
	private static final Logger log = LoggerFactory.getLogger(Recognizer.class);

	private static NativeImageLoader imageLoader = new NativeImageLoader();
	// 人脸检测窗口最小size
	private int minFaceSize = 40;

	private ComputationGraph facenet;
	private MTCNN mtcnn;

	public Recognizer() throws Exception {
		initFacenet();
		initMtcnn();
	}

	private void initFacenet() throws Exception, IOException {
		InceptionResNetV1 v1 = new InceptionResNetV1();
		v1.init();
		v1.loadWeightData();
		facenet = v1.getGraph();
	}

	private void initMtcnn() throws Exception {
		mtcnn = new MTCNN();
		mtcnn.loadModel();
	}

	/**
	 * 获取面部特征因子
	 * 
	 * @param img
	 *            源图像
	 * @param width
	 *            缩放头像宽
	 * @param height
	 *            缩放头像高
	 * @return
	 * @throws Exception
	 */
	public INDArray[] getFaceFactor(BufferedImage img, int width, int height) throws Exception {

		Vector<Box> faceBoxies = mtcnn.detectFaces(img, minFaceSize);

		if (CollectionUtils.isEmpty(faceBoxies)) {
			log.error("no face detected in image file:{}", img);
			return null;
		}

		INDArray[] output = new INDArray[faceBoxies.size()];
		for(int i=0;i<faceBoxies.size();i++){
			Box box = faceBoxies.get(i);
			INDArray face = imresample(imageLoader.asMatrix(img).get(all(), all(), interval(Math.abs(box.top()), box.top()+box.height()), interval(Math.abs(box.left()),box.left()+box.width())).dup(), height, width);
			output[i] = facenet.output(InceptionResNetV1.prewhiten(face))[1];
		}
		return output;
	}

	public Vector<Box> detectFaces(BufferedImage img) {
		return mtcnn.detectFaces(img, minFaceSize);
	}

	private INDArray imresample(INDArray img, int hs, int ws) {
		long[] shape = img.shape();
		long h = shape[2];
		long w = shape[3];
		float dx = (float) w / ws;
		float dy = (float) h / hs;
		INDArray im_data = Nd4j.create(new long[] { 1, 3, hs, ws });
		for (int a1 = 0; a1 < 3; a1++) {
			for (int a2 = 0; a2 < hs; a2++) {
				for (int a3 = 0; a3 < ws; a3++) {
					im_data.putScalar(new long[] { 0, a1, a2, a3 },
							img.getDouble(0, a1, (long) Math.floor(a2 * dy), (long) Math.floor(a3 * dx)));
				}
			}
		}
		return im_data;
	}
}
