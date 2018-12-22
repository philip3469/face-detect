package face.facenet;

import java.awt.image.BufferedImage;
import java.io.FileInputStream;

import javax.imageio.ImageIO;

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
		int width = 160;
		int height = 160;
		BufferedImage img1 = ImageIO.read(new FileInputStream(dir + "fbb1.jpg"));
		BufferedImage img2 = ImageIO.read(new FileInputStream(dir + "fbb2.jpg"));
		INDArray factor1 = recognizer.getFaceFactor(img1, width, height)[0];
		INDArray factor2 = recognizer.getFaceFactor(img2, width, height)[0];
		System.out.println(calImgLoss(factor1, factor2));
	}

	private double calImgLoss(INDArray img1, INDArray img2) {
		INDArray tmp = img1.sub(img2);
		tmp = tmp.mul(tmp).sum(1);
		return tmp.getDouble(0);
	}

}
