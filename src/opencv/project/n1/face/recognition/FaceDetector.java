/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package opencv.project.n1.face.recognition;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

/**
 *
 * @author Marlei M. Silveira
 */
public class FaceDetector {
    
    private Mat image;
    private ArrayList<Mat> detectetFaces = new ArrayList<Mat>();
    private ArrayList<Point> centerRightEye = new ArrayList<Point>(); //centro do olho direito
    private ArrayList<Point> centerLeftEye = new ArrayList<Point>(); //centro do olho esquerdo
    private int qtdFacesDetectadas; 
    
    public ArrayList<Mat> detectFaces(Mat img, String haarcascade_face, boolean printRetFace, String haarcascade_eyes, boolean printRetEyes, String haarcascade_smile, boolean printRetSmiles, String haarcascade_nose, boolean printRetNose, int alturaFaceRec, int larguraFaceRec) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        //detecta face ----------------------------------------------------------------------------------
        if (((haarcascade_face != null) && (haarcascade_face != "")) && (img != null)) {
            image = img;
            CascadeClassifier faceDetector = new CascadeClassifier(haarcascade_face);
            MatOfRect faceDetections = new MatOfRect();

            faceDetector.detectMultiScale(image, faceDetections);

            //-i-ArrayList<Mat> detectetFaces = new ArrayList<Mat>();

            if (faceDetections.toArray().length == 0) {
                System.out.println("Not Faces Detected");
            } else {
                int i = 1;
                Rect rectCrop = null;
                for (Rect rect : faceDetections.toArray()) {
                    System.out.println("Face " + i);
                    //imprime um retangulo verde sobre cada face
                    if (printRetFace == true) {
                        Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0),2);
                    }
                    //recorta apenas a face 
                    rectCrop = new Rect(rect.x, rect.y, rect.width, rect.height);
                    Mat imageCrop = new Mat(image, rectCrop);

                    //detecta olhos -------------------------------------------------------------------
                    if ((haarcascade_eyes != null) && (haarcascade_eyes != "")) {

                        CascadeClassifier eyeDetector = new CascadeClassifier(haarcascade_eyes);
                        MatOfRect eyeDetections = new MatOfRect();

                        //eyeDetector.detectMultiScale(imageCrop, eyeDetections, 1.1, 2, 0,new Size(30,30), new Size());
                        eyeDetector.detectMultiScale(imageCrop, eyeDetections);
                        if (eyeDetections.toArray().length == 0) {
                            System.out.println(" Not eyes " + i);
                        } else {
                            System.out.println("Face " + i + " with " + eyeDetections.toArray().length + " eyes");

                            Point centerREye = null; //centro do olho direito
                            Point centerLEye = null; //centro do olho esquerdo
                            int j = 1;
                            for (Rect rect1 : eyeDetections.toArray()) {
                                Point center = new Point(rect1.x + rect1.width * 0.5, rect1.y + rect1.height * 0.5);
                                //Salva a coordenada do centro dos olhos
                                if (j == 1) {
                                    centerREye = center;
                                    centerRightEye.add(center);
                                } else {
                                    if (j == 2) {
                                        centerLEye = center;
                                        centerLeftEye.add(center);
                                    }
                                }
                                j++;

                                //imprime retangulo sobre os olhos
                                if (printRetEyes == true) {
                                    Imgproc.rectangle(imageCrop, new Point(rect1.x, rect1.y), new Point(rect1.x + rect1.width, rect1.y + rect1.height),
                                            new Scalar(255, 255, 0));

                                    //imprime circulo sobre os olhos
                                    Imgproc.ellipse(imageCrop, center, new Size(rect1.width * 0.1, rect1.height * 0.1), 0, 0, 360, new Scalar(255, 0, 255), 1, 8, 0);
                                }

                            }

                            //imprime uma linha branca do centro de um olho para o outro
                            if (printRetEyes == true) {
                                if (!(centerREye == null) && !(centerLEye == null)) {
                                    Imgproc.line(imageCrop, centerREye, centerLEye, new Scalar(255, 255, 255));
                                }
                            }

                        }
                    }

                    //Detecta Sorriso/boca -------------------------------------------------------------------
                    if ((haarcascade_smile != null) && (haarcascade_smile != "")) {

                        CascadeClassifier smileDetector = new CascadeClassifier(haarcascade_smile);
                        MatOfRect smileDetections = new MatOfRect();

                        smileDetector.detectMultiScale(imageCrop, smileDetections);
                        if (smileDetections.toArray().length == 0) {
                            System.out.println(" Not smiles " + 1);
                        } else {
                            System.out.println(" Face " + i + " with " + smileDetections.toArray().length + " smiles ");
                            for (Rect rect2 : smileDetections.toArray()) {
                                //Rect rect2 = smileDetections.toArray()[0];

                                // tenta identificar o sorriso correto através de euristicas
                                // if the mouth is in the lower 2/5 of the face
                                // and the lower edge of mouth is above of the face
                                // and the horizontal center of the mouth is the enter of the face
                                Rect face = rect;
                                Rect mouth = rect2;
                                //if (mouth.y > face.y + face.height * 3 / 5 && mouth.y + mouth.height < face.y + face.height
                                //    && Math.abs((mouth.x + mouth.width / 2)) - (face.x + face.width / 2) < face.width / 10) {

                                if (mouth.y > face.height * 3 / 5 && mouth.y + mouth.height < face.height
                                        && (mouth.x + mouth.width / 2) - (face.width / 2) < face.width / 10) {
                                    //imprime retangulo sobre o sorriso
                                    if (printRetSmiles == true) {
                                        Imgproc.rectangle(imageCrop, new Point(rect2.x, rect2.y), new Point(rect2.x + rect2.height, rect2.y + rect2.width),
                                                new Scalar(0, 255, 255));
                                    }
                                }
                            }
                        }
                    }

                    //Detecta Nariz --------------------------------------------------------------------------------
                    if ((haarcascade_nose != null) && (haarcascade_nose != "")) {

                        CascadeClassifier noseDetector = new CascadeClassifier(haarcascade_nose);
                        MatOfRect noseDetections = new MatOfRect();

                        noseDetector.detectMultiScale(imageCrop, noseDetections);
                        double scaleFactor = 2;
                        int minNeighbors = 5;
                        int flags = 0;
                        Size minSize = new Size(5, 5);
                        Size maxSize = new Size(50, 50);
                        noseDetector.detectMultiScale(imageCrop, noseDetections, scaleFactor, minNeighbors, flags, minSize, maxSize);
                        //MatOfInt rejectLevels; 
                        //smileDetector.detectMultiScale3(imageCrop, smileDetections, rejectLevels, levelWeights, i, i, i, minSize, maxSize, true);            

                        if (noseDetections.toArray().length == 0) {
                            System.out.println(" Not nose " + 1);
                        } else {
                            System.out.println(" Face " + i + " with " + noseDetections.toArray().length + " noses ");
                            for (Rect rect2 : noseDetections.toArray()) {
                                // tenta identificar o nariz correto atraves de euristicas
                                // if the mouth is in the lower 2/5 of the face
                                // and the lower edge of mouth is above of the face
                                // and the horizontal center of the mouth is the enter of the face
                                Rect face = rect;
                                Rect nose = rect2;
                                //if (mouth.y > face.y + face.height * 3 / 5 && mouth.y + mouth.height < face.y + face.height
                                //    && Math.abs((mouth.x + mouth.width / 2)) - (face.x + face.width / 2) < face.width / 10) {

                                //if (nose.y >  face.height * 5 / 5 && nose.y + nose.height < face.height){
                                //  &&(nose.x + nose.width / 2) - (face.width / 2) < face.width / 10 ) {          
                                //imprime retangulo sobre o nariz
                                if (printRetNose == true) {
                                    Imgproc.rectangle(imageCrop, new Point(rect2.x, rect2.y), new Point(rect2.x + rect2.height, rect2.y + rect2.width),
                                            new Scalar(0, 0, 255));

                                    //imprime um triangulo sobre o nariz
                                    List<MatOfPoint> pts = new ArrayList();
                                    pts.add(
                                            new MatOfPoint(
                                                    new Point(rect2.x + (rect2.height / 2), rect2.y),
                                                    new Point(rect2.x + rect2.height, rect2.y + rect2.width),
                                                    new Point(rect2.x, rect2.y + rect2.width))
                                    );

                                    Imgproc.polylines(imageCrop, pts, true, new Scalar(100, 100, 100));
                                }
                                //}
                            }
                        }
                    }

                    //padroniza o tamanho da face recortada 
                    Mat resizeimage = new Mat();
                    Size sz = new Size(alturaFaceRec, larguraFaceRec);
                    Imgproc.resize(imageCrop, resizeimage, sz);

                    //Salva as faces recortadas para posteriror comparação/classificação 
                    detectetFaces.add(resizeimage);

                    String filename = "src/detectedFaces/0"+ i +" - Detected_Face.png";
                    Imgcodecs.imwrite(filename, resizeimage);

                    i++;
                }
            }
            //salva imagem com faces detectadas e destacadas com retangulos verdes
            if (printRetFace == true) {
                String filename = "src/detectedFaces/Sample_Ouput.png";
                Imgcodecs.imwrite(filename, image);
            }

            setQtdFacesDetectadas(detectetFaces.size());
            //retorna imagens das faces detectadas 
            return detectetFaces;
        } else {
            System.out.println("");
            return null;
        }
    }

    public ArrayList<Mat> getDetectetFaces() {
        return detectetFaces;
    }

    public void setDetectetFaces(ArrayList<Mat> detectetFaces) {
        this.detectetFaces = detectetFaces;
    }

    public Mat getImage() {
        return image;
    }

    public void setImage(Mat image) {
        this.image = image;
    }

    public ArrayList<Point> getCenterRightEye() {
        return centerRightEye;
    }

    public void setCenterRightEye(ArrayList<Point> centerRightEye) {
        this.centerRightEye = centerRightEye;
    }

    public ArrayList<Point> getCenterLeftEye() {
        return centerLeftEye;
    }

    public void setCenterLeftEye(ArrayList<Point> centerLeftEye) {
        this.centerLeftEye = centerLeftEye;
    }



    public int getQtdFacesDetectadas() {
        return qtdFacesDetectadas;
    }

    public void setQtdFacesDetectadas(int qtdFacesDetectadas) {
        this.qtdFacesDetectadas = qtdFacesDetectadas;
    }

}

//The detectMultiScale function is a general function that detects objects. 
//Since we are calling it on the face cascade, that’s what it detects. The first 
//option is the grayscale image.
//
//The second is the scaleFactor. Since some faces may be closer to the camera, 
//they would appear bigger than those faces in the back. The scale factor 
//compensates for this.
//
//The detection algorithm uses a moving window to detect objects. minNeighbors 
//defines how many objects are detected near the current one before it declares 
//the face found. minSize, meanwhile, gives the size of each window.
