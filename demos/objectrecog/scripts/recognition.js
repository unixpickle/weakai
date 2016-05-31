(function() {

  function findSubImage(largeImage, subImage) {
    var bigW = largeImage.width;
    var bigH = largeImage.height;
    var bigData = largeImage.getContext('2d').getImageData(0, 0, bigW, bigH).data;

    var subW = subImage.width;
    var subH = subImage.height;
    var subData = subImage.getContext('2d').getImageData(0, 0, subW, subH).data;

    var maxCorrelation = -Infinity;
    var matchX = 0;
    var matchY = 0;

    var maxX = bigW - subW;
    var maxY = bigH - subH;
    for (var y = 0; y < maxY; ++y) {
      for (var x = 0; x < maxX; ++x) {
        var correlation = 0;
        var magnitude1 = 0;
        var magnitude2 = 0;
        for (var subY = 0; subY < subH; ++subY) {
          for (var subX = 0; subX < subW; ++subX) {
            var subPixel = getImagePixel(subData, subX, subY, subW);
            var bigPixel = getImagePixel(bigData, x+subX, y+subY, bigW);
            correlation += subPixel * bigPixel;
            magnitude1 += subPixel * subPixel;
            magnitude2 += bigPixel * bigPixel;
          }
        }
        correlation /= Math.sqrt(magnitude1);
        correlation /= Math.sqrt(magnitude2);
        if (correlation > maxCorrelation) {
          maxCorrelation = correlation;
          matchX = x;
          matchY = y;
        }
      }
    }

    return {x: matchX, y: matchY};
  }

  function getImagePixel(data, x, y, w) {
    return data[4*(x+y*w)] + data[4*(x+y*w)+1] + data[4*(x+y*w)+2];
  }

  window.app.findSubImage = findSubImage;

})();
