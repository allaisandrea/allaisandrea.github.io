// const address = "Ed.Lauing@CityofPaloAlto.org;Pat.Burt@CityofPaloAlto.org;George.Lu@CityofPaloAlto.org;Julie.LythcottHaims@CityofPaloAlto.org;Keith.Reckdahl@CityofPaloAlto.org;Greer.Stone@CityofPaloAlto.org;Vicki.Veenker@CityofPaloAlto.org"
const address = "allais.andrea@gmail.com"
const cc_address = "allais.andrea@gmail.com";
const subject = "The new HVAC in Heritage Park is too loud"
const body = `Dear Councilmember,
I am a frequent user of the Heritage Park playground. The HVAC system of the newly renovated Roth building has been placed right next to the playground. It produces a constant, loud humming noise that makes for a rather unpleasant permanence. I hope that the Council can look into this nuisance: it is affecting many residents, especially children. Unless the noise is eliminated or drastically reduced, this cherished public space will be permanently degraded.
Best Regards`
let signature = ""

function set_content() {
    document.querySelector("#mailto_posted").setAttribute("href",
        "mailto:" + address +
        "?bcc=" + encodeURIComponent(cc_address) +
        "&subject=" + encodeURIComponent(subject) +
        "&body=" + encodeURIComponent(body + signature));
    document.querySelector("#mailto_anon").setAttribute("href",
        "mailto:" + address +
        "?subject=" + encodeURIComponent(subject) +
        "&body=" + encodeURIComponent(body + signature));
    document.querySelector("p.address span.value").textContent = address;
    document.querySelector("p.subject span.value").textContent = subject;
    document.querySelector("p.body span.value").textContent = body;
}

function on_signature_change(event) {
    if (event.target.value) {
        signature = ",\n" + event.target.value;
    } else {
        signature = "";
    }
    set_content();
}

document.querySelector("p.address button.clipboard").addEventListener(
    "click", () => {navigator.clipboard.writeText(address);});
document.querySelector("p.subject button.clipboard").addEventListener(
    "click", () => {navigator.clipboard.writeText(subject);});
document.querySelector("p.body button.clipboard").addEventListener(
    "click", () => {navigator.clipboard.writeText(body);});

document.querySelector("input.signature").addEventListener(
    "change", on_signature_change);

set_content();

