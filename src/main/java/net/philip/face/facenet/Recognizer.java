package net.philip.face.facenet;

import java.awt.image.BufferedImage;
import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.philip.face.facenet.model.InceptionResNetV1;
import net.philip.face.mtcnn.MTCNN;

public class Recognizer {
	private static final Logger log = LoggerFactory.getLogger(Recognizer.class);

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
	 * @param img 源图像
	 * @param width 缩放头像宽
	 * @param height 缩放头像高
	 * @return
	 * @throws Exception
	 */
	public INDArray getFaceFactor(BufferedImage img, int width, int height) throws Exception {

		INDArray[] detection = mtcnn.detect(img, 40, 160, 160);
		if (detection == null || detection.length == 0) {
			log.error("no face detected in image file:{}", img);
			return null;
		}

		if (detection.length > 1) {
			log.warn("{} faces detected in image file:{}, the first detected face will be used.", detection.length,
					img);
		}

		INDArray output[] = facenet.output(InceptionResNetV1.prewhiten(detection[0]));
		return output[1];
	}

}
