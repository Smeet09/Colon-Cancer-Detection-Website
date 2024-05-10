document.addEventListener('DOMContentLoaded', function() {
    const selectImage = document.querySelector('.select-image');
    const inputFile = document.querySelector('#file');
    const imageArea = document.querySelector('.image-area'); // Corrected selector for image-area

    if (selectImage && inputFile && imageArea) {
        selectImage.addEventListener('click', function() {
            inputFile.click();
        });

        inputFile.addEventListener('change', function() {
            const image = this.files[0];
            console.log(image);
            const reader = new FileReader();
            reader.onload = () => {
                const allImg = imageArea.querySelectorAll('img');
                allImg.forEach(item => item.remove());
                const imgUrl = reader.result;
                const img = document.createElement('img');
                img.src = imgUrl;
                imageArea.appendChild(img);
                imageArea.classList.add('active');
                imageArea.dataset.img = image.name;
            };
            reader.readAsDataURL(image); // This line was missing
        });
    } else {
        console.error("One or more required elements not found.");
    }
});