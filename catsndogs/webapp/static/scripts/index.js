var model;
const classifierElement = document.getElementById('classifier');

async function initialize() {
	model = await tf.loadLayersModel('static/model/model.json');
}

async function predict() {

	const img = document.getElementById('img');
	let tensor_img = tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
	let prediction = await model.predict(tensor_img).data();
	prediction = Math.round(prediction);

	if (prediction == 0) {
		alert("To jest kot!");
	} else if (prediction == 1) {
		alert("To jest pies!");
	}

}

function changeImage() {
	var imageDisplay = document.getElementById('img');
	var uploadedImage = document.getElementById('file_selector').files[0];
	imageDisplay.src = URL.createObjectURL(uploadedImage);
}

initialize();