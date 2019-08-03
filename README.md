# Openface_Face_Embedding-swift

## Original Image
```swift
// load face image
let face_image = UIImage(named: "jobs.jpg")!
```
![1*QFH9RIdQtjpaChl-nI59yw](https://user-images.githubusercontent.com/7419790/62411542-7b4ef100-b62f-11e9-89f8-6f033a9e2a52.jpeg)


## Face Landmark
```swift
// get face landmark
let landmark = getFaceLandmark(face_image: face_image)!
let landmark_image = getFaceLandmarkImage(image_size: face_image.size, faceLandmark: landmark)
```
<img width="696" alt="스크린샷 2019-08-03 오후 8 46 51" src="https://user-images.githubusercontent.com/7419790/62411569-ded91e80-b62f-11e9-9ced-d06b5180cc8a.png">


## Aligned Face Image
```swift
let aligned_face = getAlignedFace(face_image: face_image, faceLandmark: landmark)
```
<img width="95" alt="aligned_face" src="https://user-images.githubusercontent.com/7419790/62411594-3d9e9800-b630-11e9-8f9e-e2e0dcd20f62.png">

## Embed Face
```swift
let mlmodel = OpenFace()
let predictions = try! mlmodel.prediction(data: aligned_face.pixelBuffer(width: Int(aligned_face.size.width), height: Int(aligned_face.size.height))!)

print(predictions.output)
```
<img width="723" alt="스크린샷 2019-08-03 오후 9 30 40" src="https://user-images.githubusercontent.com/7419790/62411957-28c50300-b636-11e9-99b4-b4d7ea719e1b.png">
