package net.philip.face.mtcnn;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.Stroke;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Vector;

import javax.imageio.ImageIO;

import org.springframework.core.io.DefaultResourceLoader;

import net.philip.face.mtcnn.Box.Rect;

public class Utils {
	
	public static void main(String[] args) throws Exception {
		
		BufferedImage srcImg = ImageIO.read(new DefaultResourceLoader().getResource("bill-cook.jpg").getInputStream());
		int newWidth = 50;
		int newHeight = 50;
		BufferedImage resizeImg = new BufferedImage(newWidth, newHeight, srcImg.getType());
		Graphics2D g = resizeImg.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		g.drawImage(srcImg, 0, 0, newWidth, newHeight, 0, 0, srcImg.getWidth(), srcImg.getHeight(), null);
		g.dispose();
		ImageIO.write(resizeImg, "jpg", new File("d:/test.jpg"));
	}
	
	
	// 复制图片，并设置isMutable=true
	// public static BufferedImage copyBufferedImage(BufferedImage
	// BufferedImage){
	// return BufferedImage.copy(BufferedImage.getConfig(),true);
	// }
	// 在BufferedImage中画矩形
	// public static void drawRect(BufferedImage BufferedImage,Rect rect){
	// try {
	// Canvas canvas = new Canvas(BufferedImage);
	// Paint paint = new Paint();
	// int r=255;//(int)(Math.random()*255);
	// int g=0;//(int)(Math.random()*255);
	// int b=0;//(int)(Math.random()*255);
	// paint.setColor(Color.rgb(r, g, b));
	// paint.setStrokeWidth(1+BufferedImage.getWidth()/500 );
	// paint.setStyle(Paint.Style.STROKE);
	// canvas.drawRect(rect, paint);
	// }catch (Exception e){
	// Log.i("Utils","[*] error"+e);
	// }
	// }
	// 在图中画�?
	// public static void drawPoints(BufferedImage BufferedImage, Point[]
	// landmark){
	// for (int i=0;i<landmark.length;i++){
	// int x=landmark[i].x;
	// int y=landmark[i].y;
	// //Log.i("Utils","[*] landmarkd "+x+ " "+y);
	// drawRect(BufferedImage,new Rect(x-1,y-1,x+1,y+1));
	// }
	// }
	// Flip alone diagonal
	// 对角线翻转�?�data大小原先为h*w*stride，翻转后变成w*h*stride
	public static void flip_diag(float[] data, int h, int w, int stride) {
		float[] tmp = new float[w * h * stride];
		for (int i = 0; i < w * h * stride; i++)
			tmp[i] = data[i];
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++) {
				for (int z = 0; z < stride; z++)
					data[(x * h + y) * stride + z] = tmp[(y * w + x) * stride + z];
			}
	}

	// src转为二维存放到dst�?
	public static void expand(float[] src, float[][] dst) {
		int idx = 0;
		for (int y = 0; y < dst.length; y++)
			for (int x = 0; x < dst[0].length; x++)
				dst[y][x] = src[idx++];
	}

	// src转为三维存放到dst�?
	public static void expand(float[] src, float[][][] dst) {
		int idx = 0;
		for (int y = 0; y < dst.length; y++)
			for (int x = 0; x < dst[0].length; x++)
				for (int c = 0; c < dst[0][0].length; c++)
					dst[y][x][c] = src[idx++];

	}

	// dst=src[:,:,1]
	public static void expandProb(float[] src, float[][] dst) {
		int idx = 0;
		for (int y = 0; y < dst.length; y++)
			for (int x = 0; x < dst[0].length; x++)
				dst[y][x] = src[idx++ * 2 + 1];
	}

	// box转化为rect
	public static Rect[] boxes2rects(Vector<Box> boxes) {
		int cnt = 0;
		for (int i = 0; i < boxes.size(); i++)
			if (!boxes.get(i).deleted)
				cnt++;
		Rect[] r = new Rect[cnt];
		int idx = 0;
		for (int i = 0; i < boxes.size(); i++)
			if (!boxes.get(i).deleted)
				r[idx++] = boxes.get(i).transform2Rect();
		return r;
	}

	// 删除做了delete标记的box
	public static Vector<Box> updateBoxes(Vector<Box> boxes) {
		Vector<Box> b = new Vector<Box>();
		for (int i = 0; i < boxes.size(); i++)
			if (!boxes.get(i).deleted)
				b.addElement(boxes.get(i));
		return b;
	}

	//
	static public void showPixel(int v) {
		System.out.println("[*]Pixel:R" + ((v >> 16) & 0xff) + "G:" + ((v >> 8) & 0xff) + " B:" + (v & 0xff));
	}
	
	
	public static BufferedImage drawFaceBox(BufferedImage originalImage, Vector<Box> faceBox) {

		Graphics2D g = originalImage.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		Stroke stroke = g.getStroke();
		Color color = g.getColor();
		g.setStroke(new BasicStroke(2));
		g.setColor(Color.YELLOW);

		int radius = 1;
		for (Box box : faceBox) {
			g.drawRect(box.left(), box.top(), box.width(), box.height());
			for (Point p : box.landmark) {
				g.fillOval((int)p.getX()-radius, (int)p.getY() - radius, 2 * radius, 2 * radius);
			}
		}

		g.setStroke(stroke);
		g.setColor(color);

		g.dispose();

		return originalImage;
	}
}
