package net.philip.face;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import javax.imageio.ImageIO;
import javax.swing.WindowConstants;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
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

		// camera detect
		int camera_size = 400;
		OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);
		grabber.setImageWidth(camera_size);
		grabber.setImageHeight(camera_size);
		grabber.start();
		CanvasFrame canvas = new CanvasFrame("Camera", 1);
		canvas.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		canvas.setAlwaysOnTop(true);
		while (true) {
			if (!canvas.isVisible()) {
				break;
			}
			Frame frame = grabber.grab();
			BufferedImage image = new BufferedImage(camera_size, camera_size, BufferedImage.TYPE_3BYTE_BGR);
			Java2DFrameConverter.copy(frame, image);

			INDArray[] testFacefactors = recognizer.getFaceFactor(image, FACE_RESIZE_WIDTH, FACE_RESIZE_HEIGHT);
			
			// Vector<Box> detectFaces = recognizer.detectFaces(image);

			if (null != testFacefactors) {
				Iterator<Entry<String, INDArray>> it = faceTagMap.entrySet().iterator();
				while (it.hasNext()) {
					Entry<String, INDArray> next = it.next();
					String name = next.getKey();
					INDArray faceTagFactor = next.getValue();
					double diff = faceCompareLoss(testFacefactors[0], faceTagFactor);
					if (diff < 1.1) {
						// match
						log.info("test face is: " + name + ", face compare loss is: " + diff);
						// show detection result
						canvas.setTitle(name);
						break;
					}
				}
			}
			canvas.showImage(frame);
			Thread.sleep(50);// 图像刷新时间
		}
		grabber.stop();
		grabber.close();
	}

	private static Map<String, INDArray> loadFaceTag(String faceTagDir, Recognizer recognizer) throws Exception {
		Map<String, INDArray> faceTagMap = Maps.newHashMap();
		File dir = new File(FACE_TAG_DIR);
		if (!dir.exists()) {
			log.error("face tag dir not found!");
		}
		Collection<File> images = FileUtils.listFiles(dir, new String[] { "jpg" }, false);
		for (File image : images) {
			INDArray factor = recognizer.getFaceFactor(ImageIO.read(image), FACE_RESIZE_WIDTH, FACE_RESIZE_HEIGHT)[0];
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
