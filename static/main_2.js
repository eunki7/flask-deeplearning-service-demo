var fileDragOri = document.getElementById("file-drag-gan-ori");
var fileSelectOri = document.getElementById("file-upload-gan-ori");
var fileDragMp = document.getElementById("file-drag-gan-makeup");
var fileSelectMp = document.getElementById("file-upload-gan-makeup");

var imagePreviewOri = document.getElementById("image-preview-ori");
var imageDisplayOri = document.getElementById("image-display-ori");
var uploadCaptionOri = document.getElementById("upload-caption-ori");
var predResultOri = document.getElementById("pred-result-ori");
var loaderOri = document.getElementById("loader-ori");

var imagePreviewMakeup = document.getElementById("image-preview-makeup");
var imageDisplayMakeup = document.getElementById("image-display-makeup");
var uploadCaptionMakeup = document.getElementById("upload-caption-makeup");
var predResultMakeup = document.getElementById("pred-result-makeup");
var loaderMakeup = document.getElementById("loader-makeup");

fileDragOri.addEventListener("dragover", fileDragHoverOri, false);
fileDragOri.addEventListener("dragleave", fileDragHoverOri, false);
fileDragOri.addEventListener("drop", fileSelectHandlerOri, false);
fileSelectOri.addEventListener("change", fileSelectHandlerOri, false);

fileDragMp.addEventListener("dragover", fileDragHoverMakeup, false);
fileDragMp.addEventListener("dragleave", fileDragHoverMakeup, false);
fileDragMp.addEventListener("drop", fileSelectHandlerMakeup, false);
fileSelectMp.addEventListener("change", fileSelectHandlerMakeup, false);


document.querySelectorAll('input[type=radio][name="flexRadioDefault"]').forEach(
  radio => radio.addEventListener('change', () => 
    alert(radio.value)
  )
);

function fileDragHoverOri(e) {
  console.log(e.type);
  e.preventDefault();
  e.stopPropagation();

  fileDragOri.className = e.type === "dragover" ? "upload-box-1 dragover" : "upload-box-1";
}

function fileDragHoverMakeup(e) {
  console.log(e.type);
  e.preventDefault();
  e.stopPropagation();

  fileDragMp.className = e.type === "dragover" ? "upload-box-2 dragover" : "upload-box-2";
}

function fileSelectHandlerOri(e) {
  var files = e.target.files || e.dataTransfer.files;
  fileDragHoverOri(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFileOri(f);
  }
}

function fileSelectHandlerMakeup(e) {
  var files = e.target.files || e.dataTransfer.files;
  fileDragHoverMakeup(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFileMakeup(f);
  }
}

function previewFileOri(file) {
  console.log('previewFileOri', file.name);
  var fileName = encodeURI(file.name);

  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreviewOri.src = URL.createObjectURL(file);

    show(imagePreviewOri);
    hide(uploadCaptionOri);

    predResultOri.innerHTML = "";
    imageDisplayOri.classList.remove("loading");

    displayImage(reader.result, "image-display-ori");
  };
}

function previewFileMakeup(file) {
  console.log('previewFileMakeup', file.name);
  var fileName = encodeURI(file.name);

  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreviewMakeup.src = URL.createObjectURL(file);

    show(imagePreviewMakeup);
    hide(uploadCaptionMakeup);

    predResultMakeup.innerHTML = "";
    imageDisplayMakeup.classList.remove("loading");

    displayImage(reader.result, "image-display-makeup");
  };
}

function submitImageBeauty() {
  console.log("submit");

  if (!imageDisplayOri.src || !imageDisplayOri.src.startsWith("data")) {
    window.alert("Please select the original image.");
    return;
  }

  if (!imageDisplayMakeup.src || !imageDisplayMakeup.src.startsWith("data")) {
    window.alert("Please select a makeup image.");
    return;
  }

  loaderOri.classList.remove("hidden");
  imageDisplayOri.classList.add("loading");
  loaderMakeup.classList.remove("hidden");
  imageDisplayMakeup.classList.add("loading");

  predictImageBeauty(imageDisplayOri.src, imageDisplayMakeup.src);
}

function clearImage() {
  fileSelectOri.value = "";
  fileSelectMp.value = "";

  imagePreviewOri.src = "";
  imagePreviewMakeup.src = "";
  imageDisplayOri.src = "";
  imageDisplayMakeup.src = "";
  predResultOri.innerHTML = "";
  predResultMakeup.innerHTML = "";

  hide(imagePreviewOri);
  hide(imageDisplayOri);
  hide(loaderOri);
  hide(predResultOri);
  show(uploadCaptionOri);

  hide(imagePreviewMakeup);
  hide(imageDisplayMakeup);
  hide(loaderMakeup);
  hide(predResultMakeup);
  show(uploadCaptionMakeup);

  imageDisplayOri.classList.remove("loading");
  imageDisplayMakeup.classList.remove("loading");
}

function predictImageBeauty(oriImage, mpImage) {
  fetch("/predict-img-beauty-all", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      oriImage : oriImage,
      mpImage : mpImage
    })
  })
    .then(resp => {
      console.log(resp)
      if (resp.ok)
        resp.json().then(data => {
          displayResult(data);
        });
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
  hide(loaderOri);
  hide(loaderMakeup);
  hide(imageDisplayMakeup);
  hide(predResultMakeup);
  hide(predResultOri);

  imageDisplayOri.src = data.result
  imageDisplayOri.classList.remove("loading");
}

function hide(el) {
  el.classList.add("hidden");
}

function show(el) {
  el.classList.remove("hidden");
}