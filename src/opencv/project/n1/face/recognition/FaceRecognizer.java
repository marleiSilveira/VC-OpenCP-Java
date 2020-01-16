/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package opencv.project.n1.face.recognition;

import java.io.File;
//import java.io.FilenameFilter;
import java.nio.IntBuffer; 
import static org.bytedeco.javacpp.opencv_face.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.util.ArrayList;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author Marlei M. Silveira
 */
public class FaceRecognizer {
    
    private opencv_face.FaceRecognizer faceRecognizer;
    private ArrayList<Integer> predictedLabel = new ArrayList<Integer>(); 
    private ArrayList<String> predictedLabelInfo = new ArrayList<String>();
    private ArrayList<Double> predictedConfidence = new ArrayList<Double>(); 
    

    public FaceRecognizer(String typeRecognizer) {
        setFaceRecognizer(typeRecognizer);
    }
    
    public FaceRecognizer(String typeRecognizer, int num_components, double threshold){
        setFaceRecognizer(typeRecognizer, num_components, threshold);
    }
    
    public FaceRecognizer(String typeRecognizer, int radius, int neighbors, int grid_x, int grid_y, double threshold){
        setFaceRecognizer(typeRecognizer, radius, neighbors, grid_x, grid_y, threshold);
    }
    
    //Treina o reconhecedor de faces com imagens de referencia (com identidade conhecida)
    public void train(String trainingDir) {
        //carrega imagens de face de referencia (com identidade conhecida)
        File root = new File(trainingDir);
        File[] imageFiles = root.listFiles();

        //cria vetores de imagens de referencia e respectivos labels 
        opencv_core.MatVector images = new opencv_core.MatVector(imageFiles.length);
        opencv_core.Mat labels = new opencv_core.Mat(imageFiles.length, 1, CV_32SC1);
        IntBuffer labelsBuf = labels.getIntBuffer();

        //carrega vetores... 
        int counter = 0;
        for (File image : imageFiles) {
            //converte imagem para tons de cinza
            opencv_core.Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            
            // opencv_core.Mat img = imread(image.getAbsolutePath());

            int label = Integer.parseInt(image.getName().split("\\-")[0]);

            images.put(counter, img);

            labelsBuf.put(counter, label);
            faceRecognizer.setLabelInfo(label, image.getName());
            counter++;
        }

        //treina o reconhecedor de faces 
        faceRecognizer.train(images, labels);

    }
    
    public void recognizeFace(ArrayList <Mat> detectetFaces ){
        
        predictedLabel.clear();
        predictedLabelInfo.clear();
        predictedConfidence.clear();
        
        //String trainingDir = "src/imagens";
        
        //carrega imagem mudando o espaço de cor para tons de cinza
        //opencv_core.Mat testImage = imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);
        
        for (Mat detectetFace : detectetFaces) {
            //muda o espaço de cor para tons de cinza 
            if (detectetFace.channels() == 3) {
                Imgproc.cvtColor(detectetFace, detectetFace, Imgproc.COLOR_RGB2GRAY);
            }
            
            
            //Converte de OpenCV Mat para JavaCV Mat
            opencv_core.Mat testImage = bufferedImageToMat(matToBufferedImage(detectetFace));
 
            //faz reconhecimento da face 
            IntPointer label = new IntPointer(1);
            DoublePointer confidence = new DoublePointer(1);
            
            faceRecognizer.predict(testImage, label, confidence);  
            
            //int predictedLabel = faceRecognizer.predict(testImage);
            //int predictedLabel = faceRecognizer.predict(detectetFace);
            
            int predictedLabel = label.get(0);
            double predictedConfidence = confidence.get(0);
            
            System.out.println("Predicted label: " + predictedLabel);
            this.predictedLabel.add(label.get(0));
            
            System.out.println("Predicted label info: " + faceRecognizer.getLabelInfo(predictedLabel).getString()); 
            this.predictedLabelInfo.add(faceRecognizer.getLabelInfo(predictedLabel).getString());
            
            System.out.println("Predicted confidence: " + predictedConfidence);
            this.predictedConfidence.add(confidence.get(0));

        }
    }
    
    //Métodos para converter OpenCv Mat to JavaCV opencv_core.Mat 
    public BufferedImage matToBufferedImage(Mat frame) {       
        int type = 0;
        if (frame.channels() == 1) {
            type = BufferedImage.TYPE_BYTE_GRAY;
        } else if (frame.channels() == 3) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        BufferedImage image = new BufferedImage(frame.width() ,frame.height(), type);
        WritableRaster raster = image.getRaster();
        DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
        byte[] data = dataBuffer.getData();
        frame.get(0, 0, data);
        return image;
    }    
    
    public opencv_core.Mat bufferedImageToMat(BufferedImage bi) {
        OpenCVFrameConverter.ToMat cv = new OpenCVFrameConverter.ToMat();
        return cv.convertToMat(new Java2DFrameConverter().convert(bi)); 
    }

    
    //--------------------------------------------------------------------------
    public opencv_face.FaceRecognizer getFaceRecognizer() {
        return faceRecognizer;
    }

    public void setFaceRecognizer(String typeRecognizer){
        switch (typeRecognizer){
            case "EigenFaces" :         
                this.faceRecognizer = createEigenFaceRecognizer();
                break;        
            case "FisherFaces" : 
                this.faceRecognizer = createFisherFaceRecognizer();
                break;
            case "LBPHFaces" : 
                this.faceRecognizer = createLBPHFaceRecognizer();
                break;
            default :
                System.out.println("Type of Recognizer Invalid!");  
        }
    }
    
    public void setFaceRecognizer(String typeRecognizer, int num_components, double threshold) {
        
        switch (typeRecognizer){
            case "EigenFaces" : 
                //createEigenFaceRecognizer(int num_components=0, double threshold=DBL_MAX)
                //       - num_components – The number of components (read: Eigenfaces) kept for this Prinicpal Component Analysis. As a hint: There’s no rule how many components (read: Eigenfaces) should be kept for good reconstruction capabilities. It is based on your input data, so experiment with the number. Keeping 80 components should almost always be sufficient.
                //       - threshold – The threshold applied in the prediciton.
                this.faceRecognizer = createEigenFaceRecognizer(num_components, threshold);
                break;
            case "FisherFaces" : 
                //createFisherFaceRecognizer(int num_components=0, double threshold=DBL_MAX)
                //       - num_components – The number of components (read: Fisherfaces) kept for this Linear Discriminant Analysis with the Fisherfaces criterion. It’s useful to keep all components, that means the number of your classes c (read: subjects, persons you want to recognize). If you leave this at the default (0) or set it to a value less-equal 0 or greater (c-1), it will be set to the correct number (c-1) automatically.
                //       - The threshold applied in the prediction. If the distance to the nearest neighbor is larger than the threshold, this method returns -1    
                this.faceRecognizer = createFisherFaceRecognizer(num_components, threshold);
                break; 
            default: 
                System.out.println("Type of Recognizer Invalid!");
        }
    }
    
    public void setFaceRecognizer(String typeRecognizer, int radius, int neighbors, int grid_x, int grid_y, double threshold){
        switch (typeRecognizer){
            case "LBPHFaces" : 
                //createLBPHFaceRecognizer(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8, double threshold=DBL_MAX)
                //       - radius – The radius used for building the Circular Local Binary Pattern. The greater the radius, the
                //       - neighbors – The number of sample points to build a Circular Local Binary Pattern from. An appropriate value is to use `` 8`` sample points. Keep in mind: the more sample points you include, the higher the computational cost.
                //       - grid_x – The number of cells in the horizontal direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector.
                //       - grid_y – The number of cells in the vertical direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector.
                //       - threshold – The threshold applied in the prediction. If the distance to the nearest neighbor is larger than the threshold, this method returns -1.
                this.faceRecognizer = createLBPHFaceRecognizer(radius, neighbors, grid_x, grid_y, threshold);
                break;
            default :
                System.out.println("Type of Recognizer Invalid!");
        }
    }

    public ArrayList<Integer> getPredictedLabel() {
        return predictedLabel;
    }

    public void setPredictedLabel(ArrayList<Integer> predictedLabel) {
        this.predictedLabel = predictedLabel;
    }

    public ArrayList<String> getPredictedLabelInfo() {
        return predictedLabelInfo;
    }

    public void setPredictedLabelInfo(ArrayList<String> predictedLabelInfo) {
        this.predictedLabelInfo = predictedLabelInfo;
    }

    public ArrayList<Double> getPredictedConfidence() {
        return predictedConfidence;
    }

    public void setPredictedConfidence(ArrayList<Double> predictedConfidence) {
        this.predictedConfidence = predictedConfidence;
    }


    
}
