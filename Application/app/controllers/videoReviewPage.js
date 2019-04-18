
function back(e) {
	
	socket.off('go2ChatPage_videoRPage_response');
	$.videoReviewPage.close();
		
}

function openChat(e){

	socket.emit('go2ChatPage_videoRPage', user_id);
	numberRowsChatView = 0;
}

socket.on('go2ChatPage_videoRPage_response', function(messages){

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