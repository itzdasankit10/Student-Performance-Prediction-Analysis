document.getElementById("predictform").addEventListener("submit", async function (e) {
    e.preventDefault();

    const form = new FormData(this)

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method:"POST",
        body: form
    })

    const result = await response.json();
    document.getElementById("result-label").innerText = result.prediction
    document.getElementById("result").style.display="flex"    
})

function closeResult(){
    document.getElementById("result").style.display="none"
}