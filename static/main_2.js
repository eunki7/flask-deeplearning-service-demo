var fileDragOri = document.getElementById("file-drag-gan-ori");
var fileSelectOri = document.getElementById("file-upload-gan-ori");
var fileDragMp = document.getElementById("file-drag-gan-makeup");
var fileSelectMp = document.getElementById("file-upload-gan-makeup");

fileDragOri.addEventListener("dragover", fileDragHover, false);
fileDragOri.addEventListener("dragleave", fileDragHover, false);
fileDragOri.addEventListener("drop", fileSelectHandler, false);
fileSelectOri.addEventListener("change", fileSelectHandler, false);

fileDragMp.addEventListener("dragover", fileDragHover, false);
fileDragMp.addEventListener("dragleave", fileDragHover, false);
fileDragMp.addEventListener("drop", fileSelectHandler, false);
fileSelectMp.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  console.log(e);
  e.preventDefault();
  e.stopPropagation();

  this.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
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
    window.alert("이미지를 선택해 주세요.");
    return;
  }

  loader.classList.remove("hidden");
  imageDisplay.classList.add("loading");

  predictImageCls(imageDisplay.src);
}

function submitImageBeauty() {
  console.log("submit");

  if (!imageDisplay.src || !imageDisplay.src.startsWith("data")) {
    window.alert("이미지를 선택해 주세요.");
    return;
  }

  loader.classList.remove("hidden");
  imageDisplay.classList.add("loading");

  predictImageBeauty(imageDisplay.src);
}

function clearImage() {
  fileSelect.value = "";

  imagePreview.src = "";
  imageDisplay.src = "";
  predResult.innerHTML = "";

  hide(imagePreview);
  hide(imageDisplay);
  hide(loader);
  hide(predResult);
  show(uploadCaption);

  imageDisplay.classList.remove("loading");
}

function previewFile(file) {
  console.log(file.name);
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
  fetch("/predict-img-cls", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(image)
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {
          displayResult(data);
        });
    })
    .catch(err => {
      console.log("An error occured", err.message);
      window.alert("에러 발생!");
    });
}

function predictImageBeauty(image) {
  fetch("/predict-img-beauty", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(image)
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {
          displayResult(data);
        });
    })
    .catch(err => {
      console.log("An error occured", err.message);
      window.alert("에러 발생!");
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
  show(predResult);
}

function hide(el) {
  el.classList.add("hidden");
}

function show(el) {
  el.classList.remove("hidden");
}