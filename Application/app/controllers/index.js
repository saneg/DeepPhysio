
function showCamera (type, callback) {
	
    var camera = function () {
        // call Titanium.Media.showCamera and respond callbacks
        Ti.Media.showCamera({
            success: function (e) {
                callback(null, e);
            },
            cancel: function (e) {
            		console.log('cancel');
                callback(e, null);
            },
            error: function (e) {
                callback(e, null);
            },
            saveToPhotoGallery: true, // save our media to the gallery
            mediaTypes: [ type ],
            autohide: true,
            showControls: true,
            videoMaximumDuration: 5000
        });
        
        Titanium.Media.switchCamera(Ti.Media.CAMERA_FRONT);
    };
 
    // check if we already have permissions to capture media
    if (!Ti.Media.hasCameraPermissions()) {
 
        // request permissions to capture media
        Ti.Media.requestCameraPermissions(function (e) {
 
            // success! display the camera
            if (e.success) {
            		camera();
            		
            		var prepareDialog = Ti.UI.createAlertDialog({
                		title: 'Step Back & Get Ready',
                  	message: "A 20 seconds countdown will begin, when you'll click 'OK'",
                  	buttonNames: ['OK']
            		});
            		
            		var goDialog = Ti.UI.createAlertDialog({
                		title: 'GO',
                  	message: "Start your exercise!!!",
                  	buttonNames: ['OK']
            		});
            		
            		prepareDialog.addEventListener('click', function(e){
            			Alloy.Globals.timeleft = 20;
					Alloy.Globals.timer = setInterval(function(){
						console.log('Quaggiu: ' + Alloy.Globals.timeleft);
  						Alloy.Globals.timeleft -= 1;
  						
  						if (Alloy.Globals.timeleft == 1){
  							goDialog.show();	
  						}
  						
  						if(Alloy.Globals.timeleft <= 0){
  							goDialog.hide();
    							clearInterval(Alloy.Globals.timer);
    							Titanium.Media.startVideoCapture();
  						}
					}, 1000);
            		});
            		prepareDialog.show();           		
 
            // oops! could not obtain required permissions
            } else {
                callback(new Error('could not obtain camera permissions!'), null);
            }
        });
    } else {
        camera();
        var prepareDialog = Ti.UI.createAlertDialog({
	    		title: 'Step Back & Get Ready',
	      	message: "A 20 seconds countdown will begin, when you'll click 'OK'",
	      	buttonNames: ['OK']
		});
		
		var goDialog = Ti.UI.createAlertDialog({
	    		title: 'GO',
	      	message: "Start your exercise!!!",
	      	buttonNames: ['OK']
		});
		prepareDialog.addEventListener('click', function(e){
			Alloy.Globals.timeleft = 20;
			Alloy.Globals.timer = setInterval(function(){
				Alloy.Globals.timeleft -= 1;
				
				if (Alloy.Globals.timeleft == 1){
  					goDialog.show();	
  				}
				
				if(Alloy.Globals.timeleft <= 0){
					goDialog.hide();
					clearInterval(Alloy.Globals.timer);
					Titanium.Media.startVideoCapture();
				}
			}, 1000);
		});
		prepareDialog.show();
    }
    
}



function goToCamera(e, activitiesPage) {
	
	Alloy.Globals.noScreenAlert = true;
	
	showCamera(Ti.Media.MEDIA_TYPE_VIDEO, function (error, result) {
        if (error) {
            clearInterval(Alloy.Globals.timer);
            Alloy.Globals.timer = null;
            Alloy.Globals.timeleft = null;
            return;
        }
 
        // validate we taken a video
        if (result.mediaType == Ti.Media.MEDIA_TYPE_VIDEO) {
        	
        		socket.off('go2ListPage_response');
			socket.off('go2DonePage_response');
			socket.off('go2ChatPage_actPage_response');
			socket.off('go2VideoReviewPage_response');
			socket.off('updateActivity');
			socket.off('updateActivitiesSinceAlert');
			activitiesPage.close();
        		
        		var upPage = Alloy.createController("uploadingPage").getView();
        		var aI = upPage.getViewById('activityIndicator');
        		upPage.open();
        		aI.show();
        		
        		var progressBar = Ti.UI.createProgressBar({
    				right: '10%',
    				left: '10%',
    				height: 50,
    				min: 0,
    				max: 1,
    				value: 0,
			    top: '58%',
			    message: 'Uploading Video ...',
			    font:  {fontSize: 12, fontWeight: 'bold' },
			    color: '#000'
			});
			
			upPage.add(progressBar);
			progressBar.show();
        	
        		var xhr = Ti.Network.createHTTPClient();
        		xhr.onerror = function(e) {
            		alert(e.error);
            };
        		xhr.onload = function(e) {
        			
        			
        			var tempJSON = JSON.parse(this.responseText);
        			console.log(tempJSON);
        			console.log(tempJSON.userID);
        			
        			
            		var dialog = Ti.UI.createAlertDialog({
                		title: 'Success',
                  	message: 'The video has been completly uploaded',
                  	buttonNames: ['OK']
            		});
            		dialog.addEventListener('click', function(e){
            			var actPage = Alloy.createController("activitiesPage").getView();
            			var butt = actPage.getViewById('doneButton');
            			actPage.open();
            			butt.fireEvent('click');
            			upPage.close();
            			
            			Alloy.Globals.noScreenAlert = false;
            			if (Alloy.Globals.actionNotCorrect != null) {
            				
            				alert('The video for the activity ' + Alloy.Globals.actionNotCorrect + ' is not suitable. All the human body has to be recorded.');
            				Alloy.Globals.actionNotCorrect = null;		
            			}
            			
            			socket.emit('elaborateVideo', tempJSON.userID, tempJSON.activityName, tempJSON.videoDate);
            		});
            		dialog.show();
        		};
        		xhr.onsendstream = function(e) {
    				progressBar.value = e.progress;
    				console.log('ONSENDSTREAM - PROGRESS: ' + e.progress);
			};
        		xhr.open('POST', 'http://150.217.35.105:5000/');
        		xhr.setRequestHeader('enctype','multipart/form-data');
        		xhr.send({
            		the_video: result.media,
            		user: user_id,
            		activity: e.source.getViewById('exerciseLabel').text,
            		date: getDate()
        		}); 
        }
        
        if (result.mediaType == Ti.Media.MEDIA_TYPE_PHOTO) {
        	
        		alert("Exercise's media must be a video.");
        		return;
        }   
        
    });
}


function openActivities(e){
	
	socket.emit('logIn', $.userID_textField.value, $.password_textField.value);	
}

socket.on('logIn_response', function (resp, userID, password) {
	if (resp == 1){
		
		user_id = userID;
		user_password = password;
		
		var f = Ti.Filesystem.getFile(Ti.Filesystem.applicationDataDirectory,'user_info.json');
		if (f.exists() === false) {
    			f.createFile();
    			
    			var jsondoc = {
				'user_id': userID,
				'user_password': password
			};

			jsontext = JSON.stringify(jsondoc);

			f.write(jsontext);
		}else{
			
			var jsontext = f.read().toString();
			var jsondoc = JSON.parse(jsontext);
			
			jsondoc.user_id = userID;
			jsondoc.user_password = password;

			jsontext = JSON.stringify(jsondoc);

			f.write(jsontext);
		}
			
		
		socket.emit('openActivitiesPage', user_id);
	}else{
		
		alert('User ID or password are not correct');
	}
});

socket.on('openActivitiesPage_response', function(activities){

	var activitiesPage = Alloy.createController("activitiesPage").getView();
	var tV = activitiesPage.getViewById('tableView');
	
	for (var i=0; i<activities.length; i++){
		var newButton = Alloy.createController("components/exerciseButton");
		newButton.exerciseLabel.text = activities[i];
		newButton.row.addEventListener('click', function(e){
			goToCamera(e, activitiesPage);
		});
		tV.appendRow(newButton.row);
	}
	
	activitiesPage.open();
	socket.off('logIn_response');
	socket.off('openActivitiesPage_response');
	$.index.close();
	
});


Ti.App.addEventListener('keyboardframechanged', function(e) {
	
	$.contenitore.animate({
		top: $.contenitore.top - (Ti.Platform.displayCaps.platformHeight - e.keyboardFrame.y)/2,
	 	bottom : Math.max(0, Ti.Platform.displayCaps.platformHeight - e.keyboardFrame.y)/2,
	 	duration : 215,
	 	curve : Titanium.UI.ANIMATION_CURVE_LINEAR
 	});
 	
});



function hideKeyboard(e){
	
	$.userID_textField.blur();
	$.password_textField.blur();
	
}




$.index.open();