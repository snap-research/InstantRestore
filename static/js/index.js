window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/

    bulmaSlider.attach();

    
var videoPlayButton,
videoWrapper = document.getElementsByClassName('video-wrapper')[0],
  video = document.getElementsByTagName('video')[0],
  videoMethods = {
      renderVideoPlayButton: function() {
          if (videoWrapper.contains(video)) {
      this.formatVideoPlayButton()
              video.classList.add('has-media-controls-hidden')
              videoPlayButton = document.getElementsByClassName('video-overlay-play-button')[0]
              videoPlayButton.addEventListener('click', this.hideVideoPlayButton)
          }
      },

      formatVideoPlayButton: function() {
          videoWrapper.insertAdjacentHTML('beforeend', '\
              <svg class="video-overlay-play-button" viewBox="0 0 200 200" alt="Play video">\
                  <circle cx="100" cy="100" r="90" fill="none" stroke-width="15" stroke="#fff"/>\
                  <polygon points="70, 55 70, 145 145, 100" fill="#fff"/>\
              </svg>\
          ')
      },

      hideVideoPlayButton: function() {
          video.play()
          videoPlayButton.classList.add('is-hidden')
          video.classList.remove('has-media-controls-hidden')
          video.setAttribute('controls', 'controls')
      }
}

videoMethods.renderVideoPlayButton()

})
