$(document).ready(function () {
    $("#text-prompt-button").click(function () {
        var userPrompt = $("#user-input").val();
        $("#response-div").append("<p>You: " + userPrompt + "</p>");

        // Send the user's text prompt to the server for processing
        $.post("/process_text_prompt", { user_prompt: userPrompt }, function (data) {
            $("#response-div").append("<p>AI: " + data + "</p>");
        });

        $("#user-input").val(""); // Clear the input field
    });

    $("#voice-command-button").click(function () {
        // Trigger voice recognition and send the result to the server for processing
        $.post("/process_voice_command", {}, function (data) {
            $("#response-div").append("<p>AI: " + data + "</p>");
        });
    });
});
