document.addEventListener("DOMContentLoaded", function () {
  // Get all dropdowns
  const dropdowns = document.querySelectorAll(".drop-down");

  dropdowns.forEach(dropdown => {
    const menuLinks = dropdown.querySelectorAll(".menu ul li a");

    menuLinks.forEach(link => {
      link.addEventListener("click", function (event) {
        event.preventDefault();

        // Get the selected text
        const selectedText = this.textContent.trim();

        // Find the parent <a> element (the dropdown title)
        const parentDropdown = this.closest(".menu").previousElementSibling;

        // Update its text to show the selected option
        if (parentDropdown) {
          const baseText = parentDropdown.textContent.split(":")[0].trim();
          parentDropdown.innerHTML = `${baseText}: <strong>${selectedText}</strong> <i class="fas fa-caret-down"></i>`;
        }
      });
    });
  });
});
