document.addEventListener("DOMContentLoaded", function () {
    const trainingForm = document.getElementById("training-form");
    const resultsSection = document.getElementById("results");
    const lossAccuracyChart = document.getElementById("loss-accuracy-chart");
    const downloadButton = document.getElementById("download-button");

    trainingForm.addEventListener("submit", function (event) {
        event.preventDefault();

        // Create a FormData object to hold the form data
        const formData = new FormData(trainingForm);

        // Make an AJAX request to the Flask endpoint
        fetch('/train', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(results => {
            // Update loss-accuracy chart (using appropriate charting library)
            // Enable download button
            downloadButton.setAttribute("href", "path_to_your_trained_model.zip");
        });
    });
});




// document.addEventListener("DOMContentLoaded", function () {
//     const trainingForm = document.getElementById("training-form");
//     const resultsSection = document.getElementById("results");
//     const lossAccuracyChart = document.getElementById("loss-accuracy-chart");
//     const downloadButton = document.getElementById("download-button");

//     trainingForm.addEventListener("submit", function (event) {
//         event.preventDefault();
//         // Inside the trainingForm.addEventListener("submit", function (event) { ... })

// // Make an AJAX request to the Flask endpoint
// fetch('/submit', {
//     method: 'POST',
//     body: new FormData(trainingForm)
// })
// .then(response => response.json())
// .then(results => {
//     // Update loss-accuracy chart (using appropriate charting library)
//     // Enable download button
//     downloadButton.setAttribute("href", "path_to_your_trained_model.zip");
// });


//         // Simulate training and generate some example data
//         const trainingData = [0.8, 0.6, 0.4, 0.2, 0.1];
//         const validationData = [0.7, 0.5, 0.3, 0.2, 0.15];

//         // Display results section
//         resultsSection.style.display = "block";

//         // Update loss-accuracy chart
//         // You can use libraries like Chart.js to create charts

//         // Enable download button
//         downloadButton.setAttribute("href", "path_to_your_trained_model.zip");
//     });
// });
