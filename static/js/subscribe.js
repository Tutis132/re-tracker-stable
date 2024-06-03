document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('subscription-form');
    const emailInput = document.getElementById('email');
    const messageParagraph = document.getElementById('subscription-message');

    form.addEventListener('submit', function(event) {
        event.preventDefault(); 

        if (emailInput.value) {
            messageParagraph.textContent = 'Thank you for subscribing!';
            emailInput.value = ''; 
        } else {
            messageParagraph.textContent = 'Please enter a valid email address.';
        }
    });
});
