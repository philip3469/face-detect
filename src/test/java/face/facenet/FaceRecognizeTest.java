package face.facenet;

import java.awt.image.BufferedImage;
import java.io.FileInputStream;

import javax.imageio.ImageIO;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import net.philip.face.facenet.Recognizer;

public class FaceRecognizeTest {
	private static Recognizer recognizer;
	private String dir = "src/test/resources/images/facenet/";

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		recognizer = new Recognizer();
	}

	@Test
	public void cutFaceAndCompareTest() throws Exception {
		BufferedImage img1 = ImageIO.read(new FileInputStream(dir + "fbb1.jpg"));
		BufferedImage img2 = ImageIO.read(new FileInputStream(dir + "fbb2.jpg"));
		INDArray factor1 = recognizer.getFaceFactor(img1)[0];
		INDArray factor2 = recognizer.getFaceFactor(img2)[0];

		// face euclidean distance
		double distance = factor1.distance2(factor2);
		System.out.println("distance is " + distance);
		Assert.assertTrue(distance < 0.6);
	}

}
