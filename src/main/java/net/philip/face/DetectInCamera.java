package net.philip.face;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import javax.imageio.ImageIO;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Maps;

import net.philip.face.facenet.Recognizer;

public class DetectInCamera {
	private static final Logger log = LoggerFactory.getLogger(DetectInCamera.class);

	private static final String FACE_TAG_DIR = "d:/face_tag";
	private static final int FACE_RESIZE_WIDTH = 160;
	private static final int FACE_RESIZE_HEIGHT = 160;

	public static void main(String[] args) throws Exception {

		Recognizer recognizer = new Recognizer();
		// load face tag map with factor
		Map<String, INDArray> faceTagMap = loadFaceTag(FACE_TAG_DIR, recognizer);

		// test
		BufferedImage input = ImageIO.read(new File("d:/14.jpg"));
		INDArray testFaceFactor = recognizer.getFaceFactor(input, FACE_RESIZE_WIDTH, FACE_RESIZE_HEIGHT);

		Iterator<Entry<String, INDArray>> it = faceTagMap.entrySet().iterator();
		while (it.hasNext()) {
			Entry<String, INDArray> next = it.next();
			String name = next.getKey();
			INDArray faceTagFactor = next.getValue();
			double diff = faceCompareLoss(testFaceFactor, faceTagFactor);
			if (diff < 1.1) {
				// match
				log.info("test face is: " + name + ", face compare loss is: " + diff);
				return;
			}
		}

		log.info("not match");
	}

	private static Map<String, INDArray> loadFaceTag(String faceTagDir, Recognizer recognizer) throws Exception {
		Map<String, INDArray> faceTagMap = Maps.newHashMap();
		File dir = new File(FACE_TAG_DIR);
		if (!dir.exists()) {
			log.error("face tag dir not found!");
		}
		Collection<File> images = FileUtils.listFiles(dir, new String[] { "jpg" }, false);
		for (File image : images) {
			INDArray factor = recognizer.getFaceFactor(ImageIO.read(image), FACE_RESIZE_WIDTH, FACE_RESIZE_HEIGHT);
			String name = StringUtils.substringBeforeLast(image.getName(), ".");
			faceTagMap.put(name, factor);
			log.info("tag face loaded: {}", image.getName());
		}
		return faceTagMap;
	}

	private static double faceCompareLoss(INDArray face1, INDArray face2) {
		INDArray tmp = face1.sub(face2);
		tmp = tmp.mul(tmp).sum(1);
		return tmp.getDouble(0);
	}

}
