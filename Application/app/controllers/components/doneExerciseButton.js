function showCamera (type, callback) {
    var camera = function () {
        // call Titanium.Media.showCamera and respond callbacks
        Ti.Media.showCamera({
            success: function (e) {
                callback(null, e);
            },
            cancel: function (e) {
                callback(e, null);
            },
            error: function (e) {
                callback(e, null);
            },
            saveToPhotoGallery: true, // save our media to the gallery
            mediaTypes: [ type ]
        });
    };
 
    // check if we already have permissions to capture media
    if (!Ti.Media.hasCameraPermissions()) {
 
        // request permissions to capture media
        Ti.Media.requestCameraPermissions(function (e) {
 
            // success! display the camera
            if (e.success) {
                camera();
 
            // oops! could not obtain required permissions
            } else {
                callback(new Error('could not obtain camera permissions!'), null);
            }
        });
    } else {
        camera();
    }
}



function doClick(e) {
	
	if($.NNResult_image.image != null){
		
		socket.emit('go2VideoReviewPage', user_id, $.exerciseLabel.text);
			
    	}
}


