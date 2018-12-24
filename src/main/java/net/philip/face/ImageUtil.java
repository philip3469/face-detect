package net.philip.face;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;

import javax.swing.WindowConstants;

import org.bytedeco.javacv.CanvasFrame;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ImageUtil {

	/**
	 * show frame in canvas
	 * 
	 * @param face
	 */
	public static void showFrame(INDArray image) {
		CanvasFrame canvas = new CanvasFrame("Debug Frame", 1);
		canvas.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		canvas.setAlwaysOnTop(true);
		canvas.showImage(toImage(image));
	}

	/**
	 * ndarray(0,3,height,width) to image
	 * 
	 * @param image
	 * @return
	 */
	public static BufferedImage toImage(INDArray image) {
		long[] shape = image.shape();
		BufferedImage bImg = new BufferedImage((int) shape[3], (int) shape[2], BufferedImage.TYPE_INT_RGB);
		WritableRaster raster = bImg.getRaster();
		for (int i = 0; i < bImg.getWidth(); i++) {
			for (int j = 0; j < bImg.getHeight(); j++) {
				raster.setPixel(i, j, new double[] { image.getDouble(0, 2, j, i), image.getDouble(0, 1, j, i),
						image.getDouble(0, 0, j, i) });
			}
		}
		return bImg;
	}

}
