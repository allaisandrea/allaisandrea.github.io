// Comment
img_element = document.querySelector("#img-slope-distribution")
let img_slope_distribution_index = 0
setInterval(() => {
    img_slope_distribution_index = (img_slope_distribution_index + 1) % 4;
    img_element.src = `/pages/diffusion/slope_distribution_${img_slope_distribution_index}.svg`;
}, 2000);
