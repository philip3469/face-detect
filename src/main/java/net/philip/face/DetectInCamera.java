package net.philip.face;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Vector;

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
import net.philip.face.mtcnn.Box;

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
		int camera_size = 800;
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

			Vector<Box> detectFaces = recognizer.detectFaces(image);
			for (Box box : detectFaces) {
				INDArray factor = recognizer.getFaceFactor(box, image, FACE_RESIZE_WIDTH, FACE_RESIZE_HEIGHT);
				
				double minLoss = 1;
				String minLossName = null;
				Iterator<Entry<String, INDArray>> it = faceTagMap.entrySet().iterator();
				while (it.hasNext()) {
					Entry<String, INDArray> next = it.next();
					String name = next.getKey();
					INDArray faceTagFactor = next.getValue();
					double diff = faceCompareLoss(factor, faceTagFactor);
					if (diff < minLoss) {
						minLoss=diff;
						minLossName=name;
					}
				}
		
				if(null != minLossName){
					// match
					log.info("test face is: " + minLossName + ", face compare loss is: " + minLoss);
					// show detection result
					Graphics g = image.getGraphics();
					g.setColor(Color.RED);
					g.drawRect(box.left(), box.top(), box.width(),
							box.height());
					g.drawString(minLossName, box.left() + 5, box.top() + 15);
					Java2DFrameConverter.copy(image, frame);
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
			INDArray[] faceFactors = recognizer.getFaceFactor(ImageIO.read(image), FACE_RESIZE_WIDTH,
					FACE_RESIZE_HEIGHT);
			if (null == faceFactors) {
				log.error("no face dected in {}", image.getName());
				continue;
			}
			String name = StringUtils.substringBeforeLast(image.getName(), ".");
			faceTagMap.put(name, faceFactors[0]);
			log.info("tag face loaded: {}", image.getName());
		}
		return faceTagMap;
	}

	//euclidean distance
	private static double faceCompareLoss(INDArray face1, INDArray face2) {
		return face1.distance2(face2);
	}

}
