// static/js/script.js
async function Prediction_sentiment() {
    const inputData = document.getElementById('inputData').value;
    const response_sentiment = await fetch('/Prediction_sentiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: inputData })
    });
    result_sentiment = await response_sentiment.text();
    result_sentiment = result_sentiment.replace(/^"|"$/g, '');
    document.getElementById('inputData_sentiment').value = result_sentiment;
}
async function Prediction_emotion() {
    const inputData = document.getElementById('inputData').value;
    const response_emotion = await fetch('/Prediction_emotion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: inputData })
    });
    result_emotion = await response_emotion.text();
    result_emotion = result_emotion.replace(/^"|"$/g, '');
    document.getElementById('inputData_emotion').value = result_emotion;
}
async function Prediction_Offensive_Language() {
    const inputData = document.getElementById('inputData').value;
    const response_Offensive_Language = await fetch('/Prediction_Offensive_Language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: inputData })
    });
    result_ROL = await response_Offensive_Language.text();
    result_ROL = result_ROL.replace(/^"|"$/g, '');
    document.getElementById('inputData_Offensive_Language').value = result_ROL;
}
async function Clear_all() {
    document.getElementById("inputData").value="";
    document.getElementById('inputData_sentiment').value = "";
    document.getElementById('inputData_emotion').value = "";
    document.getElementById('inputData_Offensive_Language').value = "";
}