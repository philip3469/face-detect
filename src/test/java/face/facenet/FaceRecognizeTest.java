package face.facenet;

import java.awt.image.BufferedImage;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import net.philip.face.ImageUtil;
import net.philip.face.facenet.Recognizer;

public class FaceRecognizeTest {
	private static Recognizer recognizer;
	private String dir = "src/test/resources/images";

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		recognizer = new Recognizer();
	}

	@Test
	public void cutFaceAndCompareTest() throws Exception {
		BufferedImage img1 = ImageUtil.loadImage(dir + "/facenet/fbb1.jpg");
		BufferedImage img2 = ImageUtil.loadImage(dir + "/facenet/fbb2.jpg");
		INDArray factor1 = recognizer.getFaceFactor(img1)[0];
		INDArray factor2 = recognizer.getFaceFactor(img2)[0];

		// face euclidean distance
		double distance = ImageUtil.faceEuclideanDistance(factor1, factor2);
		System.out.println("distance is " + distance);
		Assert.assertTrue(distance < 0.6);
	}

	@Test
	public void getFaceFactor() throws Exception{
		BufferedImage image = ImageUtil.loadImage(dir + "/pivotal-ipo-nyse.jpg");
		INDArray[] factors = recognizer.getFaceFactor(image);
		Assert.assertEquals(8, factors.length);
	}

}
