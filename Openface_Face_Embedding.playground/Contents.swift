import UIKit
import CoreML



let face_image = UIImage(named: "jobs.jpg")!


let landmark = getFaceLandmark(face_image: face_image)!
let landmark_image = getFaceLandmarkImage(image_size: face_image.size, faceLandmark: landmark)

let aligned_face = getAlignedFace(face_image: face_image, faceLandmark: landmark)!


let mlmodel = OpenFace()
let predictions = try! mlmodel.prediction(data: aligned_face.pixelBuffer(width: Int(aligned_face.size.width), height: Int(aligned_face.size.height))!)

print(predictions.output)

