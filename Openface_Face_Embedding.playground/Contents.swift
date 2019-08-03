import UIKit


let face_image = UIImage(named: "jobs.jpg")!


let landmark = getFaceLandmark(face_image: face_image)!
let landmark_image = getFaceLandmarkImage(image_size: face_image.size, faceLandmark: landmark)

let aligned_face = getAlignedFace(face_image: face_image, faceLandmark: landmark)
