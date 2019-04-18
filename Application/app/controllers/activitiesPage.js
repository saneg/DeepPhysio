
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



function goToCamera(e) {
	
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
			$.activitiesPage.close();
        		
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


function goToRetryCamera(e, videoReviewPage) {
	
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
			$.activitiesPage.close();
			
			socket.off('go2ChatPage_videoRPage_response');
			videoReviewPage.close();
        		
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
            		activity: videoReviewPage.getViewById('videoPlayer').representedActivity,
            		date: getDate()
        		}); 
        }
        
        if (result.mediaType == Ti.Media.MEDIA_TYPE_PHOTO) {
        	
        		alert("Exercise's media must be a video.");
        		return;
        }
       
    });
}








function go2ListPage(e){
	
	socket.emit('go2ListPage', user_id);
}


socket.on('go2ListPage_response', function (activities) {
	
	tV = $.activitiesPage.getViewById('tableView'); 
	$.activitiesPage.remove(tV);
	$.doneButton_image.image = "/images/doneButton.png";
	$.doneButton.enabled = true;
	$.listButton_image.image = "/images/listButton_pressed.png";
	$.listButton.enabled = false;
	
	
	var tableView = Ti.UI.createTableView({
		id: 'tableView',
		width: Ti.UI.FILL,
		height: '80.3%',
		top: '11.5%',
		separatorStyle: "none",
		selectionStyle: "none"
	});
	
	for (var i=0; i<activities.length; i++){
		var newButton = Alloy.createController("components/exerciseButton");
		newButton.exerciseLabel.text = activities[i];
		newButton.row.addEventListener('click', function(e){
			goToCamera(e);
		});
		tableView.appendRow(newButton.row);
	}
	
	
	$.activitiesPage.add(tableView);
});




function go2DonePage(e){
	
	socket.emit('go2DonePage', user_id);
}


socket.on('go2DonePage_response', function (activities, status) {
	
	tV = $.activitiesPage.getViewById('tableView'); 
	$.activitiesPage.remove(tV);
	$.listButton_image.image = "/images/listButton.png";
	$.listButton.enabled = true;
	$.doneButton_image.image = "/images/doneButton_pressed.png";
	$.doneButton.enabled = false;
	
	var tableView = Ti.UI.createTableView({
		id: 'tableView',
		width: Ti.UI.FILL,
		height: '80.3%',
		top: '11.5%',
		separatorStyle: "none",
		selectionStyle: "none"
	});
	
	
	for (var i=0; i<activities.length; i++){
		var newB = Alloy.createController("components/doneExerciseButton");
		newB.exerciseLabel.text = activities[i];
		if(status[i] == 1){
			newB.NNResult_image.image = "/images/doneButtonIcon.png";
		}else{
			if(status[i] == 0){
				newB.NNResult_image.image = "/images/wrongButtonIcon.png";	
			}
		}
		tableView.appendRow(newB.row);
	}
	
	$.activitiesPage.add(tableView);
});


socket.on('updateActivity', function(activity_name, resp){
	tableV = $.activitiesPage.getViewById('tableView');
	rows = tableV.data[0].rows;
	for (var i=0; i<rows.length; i++){
		exLabel = rows[i].getViewById('exerciseLabel');
		if(exLabel.text == activity_name){
			NNimage = rows[i].getViewById('NNResult_image');
			if(resp == 1){
				NNimage.image = "/images/doneButtonIcon.png";
			}else{
				if(resp == 0){
					NNimage.image = "/images/wrongButtonIcon.png";	
				}
			}
			break;		
		}
	}	
});




function openChat(e){
		
	socket.emit('go2ChatPage_actPage', user_id);
	numberRowsChatView = 0;
}

socket.on('go2ChatPage_actPage_response', function(messages){

	var chatPage = Alloy.createController("chatPage").getView();
	var tA = chatPage.getViewById('tableArea');
	
	for (var i=0; i<messages.length; i++){
		if(i==0){
			var newL = Alloy.createController("components/dateLabelChat");
			var messageDate = messages[i].sendDate;
			newL.date_label.text = "-" + messageDate.split(" ")[0] + "-";
			tA.appendRow(newL.row);
			numberRowsChatView = numberRowsChatView + 1;	
			actualDateChat = messageDate.split(" ")[0];
		}else{
			var messageDate = messages[i].sendDate;
			if(actualDateChat != messageDate.split(" ")[0]){
				var newL = Alloy.createController("components/dateLabelChat");
				newL.date_label.text = "-" + messageDate.split(" ")[0] + "-";
				tA.appendRow(newL.row);
				numberRowsChatView = numberRowsChatView + 1;		
				actualDateChat = messageDate.split(" ")[0];
			}
		}
		
		if(messages[i].senderID == user_id){
			var newBal = Alloy.createController("components/myMessageView");
			newBal.myMessage.text = messages[i].message;
			tA.appendRow(newBal.row);	
			numberRowsChatView = numberRowsChatView + 1;	
		}else{
			var newBal = Alloy.createController("components/otherMessageView");
			newBal.otherMessage.text = messages[i].message;
			tA.appendRow(newBal.row);
			numberRowsChatView = numberRowsChatView + 1;	
		}
 		
	}
	
	chatPage.open();
	tA.scrollToIndex(numberRowsChatView-1);
	
});


socket.on('go2VideoReviewPage_response', function(videoResults){	
	
	var vRP = Alloy.createController("videoReviewPage").getView();
	
	var retryB = vRP.getViewById('retryButton');
	retryB.addEventListener('click', function(e){
		goToRetryCamera(e, vRP);
	});
	
	var vP = vRP.getViewById('videoPlayer');
	vP.url = "http://150.217.35.105:5000" + videoResults['videoPath'];
	
	vP.representedActivity = videoResults['activity'];
	
	
	var daEs = vRP.getViewById('daEseguire');


	daEs.attributedString = Ti.UI.createAttributedString({
		text: "Exercise to perform: " + videoResults['activity'],
		attributes: [
    			{
        			type: Ti.UI.ATTRIBUTE_FONT,
            		value: {fontWeight: 'bold'},
            		range: [0, 19]
        		}
    		]
	});
	
	var eseg = vRP.getViewById('eseguito');
	
	if (videoResults['check'] == 1){
		
		if (videoResults['ratio'] >= 0.7){
			
			eseg.attributedString = Ti.UI.createAttributedString({
				text: "Exercise performed: " + videoResults['prediction'] + "",
				attributes: [
    					{
        					type: Ti.UI.ATTRIBUTE_FONT,
            				value: {fontWeight: 'bold'},
            				range: [0, 18]
        				},
        				{
        					type: Ti.UI.ATTRIBUTE_FOREGROUND_COLOR,
            				value: '#e3ac00',
            				range: [20, videoResults['prediction'].length]
        				}
    				]
			});
		}else{
			
			eseg.attributedString = Ti.UI.createAttributedString({
				text: "Exercise performed: " + videoResults['prediction'] + "",
				attributes: [
    					{
        					type: Ti.UI.ATTRIBUTE_FONT,
            				value: {fontWeight: 'bold'},
            				range: [0, 18]
        				},
        				{
        					type: Ti.UI.ATTRIBUTE_FOREGROUND_COLOR,
            				value: '#0c0',
            				range: [20, videoResults['prediction'].length]
        				}
    				]
			});	
		}
		
	}else{
		
		eseg.attributedString = Ti.UI.createAttributedString({
			text: "Exercise performed: " + videoResults['prediction'] + "",
			attributes: [
    				{
        				type: Ti.UI.ATTRIBUTE_FONT,
            			value: {fontWeight: 'bold'},
            			range: [0, 18]
        			},
        			{
        				type: Ti.UI.ATTRIBUTE_FOREGROUND_COLOR,
            			value: '#cc0000',
            			range: [20, videoResults['prediction'].length]
        			}
    			]
		});
		
	}
	
	
	
	vRP.open();
	
});




socket.on('updateActivitiesSinceAlert', function(activity_name){
	
	tableV = $.activitiesPage.getViewById('tableView');
	rows = tableV.data[0].rows;
	found = false;
	for (var i=0; i<rows.length; i++){
		exLabel = rows[i].getViewById('exerciseLabel');
		if(exLabel.text == activity_name){
			found = true;
			tableV.deleteRow(rows[i]);
			break;		
		}
	}
	
	if (found == false){
		var newButton = Alloy.createController("components/exerciseButton");
		newButton.exerciseLabel.text = activity_name;
		newButton.row.addEventListener('click', function(e){
			goToCamera(e);
		});
		tableV.appendRow(newButton.row);
	}
	
	
});





