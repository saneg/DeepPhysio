
var args = $.args;
 
Ti.App.addEventListener('keyboardframechanged', function(e) {
	
	$.contenitore.animate({
		top: $.contenitore.top - (Ti.Platform.displayCaps.platformHeight - e.keyboardFrame.y),
	 	bottom : Math.max(0, Ti.Platform.displayCaps.platformHeight - e.keyboardFrame.y),
	 	duration : 215,
	 	curve : Titanium.UI.ANIMATION_CURVE_LINEAR
 	});
 	
 	
 	if ((Ti.Platform.displayCaps.platformHeight - e.keyboardFrame.y) == 0){
 		
 		$.tableArea.animate({
			top: '11.5%',
			height: '80.3%',
	 		duration : 215,
	 		curve : Titanium.UI.ANIMATION_CURVE_LINEAR
 		});	
 		
 	}else{
 		
 		if (Ti.Platform.displayCaps.platformHeight < 700 || Ti.Platform.displayCaps.platformHeight > 870){
 			
 			$.tableArea.animate({
				top: '50.4%',
				height: '41.4%',
	 			duration : 215,
	 			curve : Titanium.UI.ANIMATION_CURVE_LINEAR
 			});	
 		}else{
 			
 			if (Ti.Platform.displayCaps.platformHeight < 800){
 				
 				$.tableArea.animate({
					top: '48.4%',
					height: '42.4%',
	 				duration : 215,
	 				curve : Titanium.UI.ANIMATION_CURVE_LINEAR
 				});
 				
 			}else{
 				
 				$.tableArea.animate({
					top: '52.4%',
					height: '39.4%',
	 				duration : 215,
	 				curve : Titanium.UI.ANIMATION_CURVE_LINEAR
 				});
 				
 				
 			}
 			
 		}
 	
 	}
 	
});


function sendMessage(e){
	
	if($.textMessage.value != ''){
		
		socket.emit('sendMessage', user_id, $.textMessage.value, getDate());
		$.textMessage.value = '';	
	}
}


socket.on('newMessage', function(message){
	
	var messageDate = message.sendDate;
	if(actualDateChat != messageDate.split(" ")[0]){
		var newL = Alloy.createController("components/dateLabelChat");
		newL.date_label.text = "-" + messageDate.split(" ")[0] + "-";
		$.tableArea.appendRow(newL.row);	
		numberRowsChatView = numberRowsChatView + 1;
		actualDateChat = messageDate.split(" ")[0];
	}
	
	if(message.senderID == user_id){
		var newBal = Alloy.createController("components/myMessageView");
		newBal.myMessage.text = message.message;
		$.tableArea.appendRow(newBal.row);	
		numberRowsChatView = numberRowsChatView + 1;
	}else{
		var newBal = Alloy.createController("components/otherMessageView");
		newBal.otherMessage.text = message.message;
		$.tableArea.appendRow(newBal.row);
		numberRowsChatView = numberRowsChatView + 1;
	}

	$.tableArea.scrollToIndex(numberRowsChatView-1);
	
});



function hideKeyboard(e){
	
	$.textMessage.blur();	
}





function back(e) {
	
	socket.off('newMessage');
	$.chatPage.close();		
}