var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

var submitBtn = document.getElementById("submit");

function fileDragHover(e) {
  e.preventDefault();
  e.stopPropagation();

  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  var files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFile(f);
  }
}

var imagePreview = document.getElementById("image-preview");
var imageDisplay = document.getElementById("image-display");
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result");
var loader = document.getElementById("loader");

function submitImageCls() {

  if (!imageDisplay.src || !imageDisplay.src.startsWith("data")) {
    window.alert("Please select a image.");
    return;
  }

  loader.classList.remove("hidden");
  imageDisplay.classList.add("loading");

  predictImageCls(imageDisplay.src);
}

function clearImage() {
  fileSelect.value = "";

  imagePreview.src = "";
  imageDisplay.src = "";
  predResult.innerHTML = "";

  submitBtn.disabled = false

  hide(imagePreview);
  hide(imageDisplay);
  hide(loader);
  hide(predResult);
  show(uploadCaption);

  imageDisplay.classList.remove("loading");
}

function previewFile(file) {
  var fileName = encodeURI(file.name);

  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreview.src = URL.createObjectURL(file);

    show(imagePreview);
    hide(uploadCaption);

    predResult.innerHTML = "";
    imageDisplay.classList.remove("loading");

    displayImage(reader.result, "image-display");
  };
}

function predictImageCls(image) {
  fetch("/predict/img-cls", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ oriImage : image })
  })
    .then(resp => {
      if (resp.ok) {
        resp.json().then(data => {
          displayResult(data);
        });
      } else {
        clearImage()
      }
    })
    .catch(err => {
      console.log("An error occured", err.message);
    });
}

function displayImage(image, id) {
  let display = document.getElementById(id);
  display.src = image;
  show(display);
}

function displayResult(data) {
  hide(loader);
  predResult.innerHTML = data.result;
  submitBtn.disabled = true
  show(predResult);
}

function hide(el) {
  el.classList.add("hidden");
}

function show(el) {
  el.classList.remove("hidden");
}