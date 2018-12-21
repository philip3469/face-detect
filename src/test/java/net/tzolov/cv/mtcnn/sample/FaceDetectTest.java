package net.tzolov.cv.mtcnn.sample;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Vector;

import javax.imageio.ImageIO;

import org.springframework.core.io.DefaultResourceLoader;

import com.fasterxml.jackson.databind.ObjectMapper;

import net.philip.face.detect.Box;
import net.philip.face.detect.MTCNN;
import net.philip.face.detect.Utils;

public class FaceDetectTest {

	public static void main(String[] args) throws Exception {

		MTCNN mtcnn = new MTCNN();
		mtcnn.loadModel();

		String testImgName = "images/test.jpg";
		BufferedImage srcImg = ImageIO.read(new DefaultResourceLoader().getResource(testImgName).getInputStream());

		long start = System.currentTimeMillis();
		Vector<Box> detectFaces = mtcnn.detectFaces(srcImg, 40);
		System.out.println("########### cost(ms): " + (System.currentTimeMillis() - start)); // avg=337ms

		String json = new ObjectMapper().writeValueAsString(detectFaces);
		System.out.println(json);

		BufferedImage result = Utils.drawFaceBox(srcImg, detectFaces);
		ImageIO.write(result, "jpg", new File("src/test/resources/images/test/face-detect-test.jpg"));
	}

}
