const address = "allais.andrea@gmail.com;pame.milani@gmail.com"
const subject = "Subject"
const body = "Body"
document.querySelector("a.mailto").setAttribute("href", "mailto:" + address +
    "?subject=" + encodeURIComponent(subject) +
    "&body=" + encodeURIComponent(body));
document.querySelector("p.address span.value").textContent = address
document.querySelector("p.subject span.value").textContent = subject
document.querySelector("p.body span.value").textContent = body


