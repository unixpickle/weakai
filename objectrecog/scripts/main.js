(function() {

  var camera = null;
  var controls = null;
  var canvas = null;

  function initialize() {
    camera = new window.app.Camera();
    controls = new window.app.Controls();
    canvas = document.getElementById('canvas');

    camera.on('error', function() {
      document.body.innerHTML = '<div id="error">Camera API not supported</div>';
    });

    camera.on('start', function(dimensions) {
      controls.enable();
      canvas.width = dimensions.width;
      canvas.height = dimensions.height;
      canvas.style.left = 'calc(50% - ' + Math.round(dimensions.width/2) + 'px)';
    });

    camera.on('frame', function(frame) {
      canvas.getContext('2d').drawImage(frame, 0, 0);
    });

    camera.start();
  }

  window.addEventListener('load', initialize);

})();
