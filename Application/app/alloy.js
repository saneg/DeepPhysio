// The contents of this file will be executed before any of
// your view controllers are ever executed, including the index.
// You have access to all functionality on the `Alloy` namespace.
//
// This is a great place to do any initialization for your app
// or create any global variables/functions that you'd like to
// make available throughout your app. You can easily make things
// accessible globally by attaching them to the `Alloy.Globals`
// object. For example:
//
// Alloy.Globals.someGlobalFunction = function(){};

// added during app creation. this will automatically login to
// ACS for your application and then fire an event (see below)
// when connected or errored. if you do not use ACS in your
// application as a client, you should remove this block


//(function(){
//var ACS = require('ti.cloud'),
//    env = Ti.App.deployType.toLowerCase() === 'production' ? 'production' : 'development',
//    username = Ti.App.Properties.getString('acs-username-'+env),
//    password = Ti.App.Properties.getString('acs-password-'+env);
//
// if not configured, just return
//if (!env || !username || !password) { return; }
///**
// * Appcelerator Cloud (ACS) Admin User Login Logic
// *
// * fires login.success with the user as argument on success
// * fires login.failed with the result as argument on error
// */
//ACS.Users.login({
//	login:username,
//	password:password,
//}, function(result){
//	if (env==='development') {
//		Ti.API.info('ACS Login Results for environment `'+env+'`:');
//		Ti.API.info(result);
//	}
//	if (result && result.success && result.users && result.users.length){
//		Ti.App.fireEvent('login.success',result.users[0],env);
//	} else {
//		Ti.App.fireEvent('login.failed',result,env);
//	}
//});
//
//})();


var user_id = null;
var user_password = null;

Alloy.Globals.timer = null;
Alloy.Globals.timeleft = null;




var io = require("ti.socketio");	

	
var socket = io.connect('http://150.217.35.105:5000/');
socket.on('connect', function () {
	console.log('connected');

	var file = Titanium.Filesystem.getFile(Titanium.Filesystem.applicationDataDirectory, 'user_info.json'); 
	
	if (file.exists() === true) {
	
		var jtext = file.read().toString();
		var jdoc = JSON.parse(jtext);
	
		socket.emit('update_userSocketID', jdoc.user_id, jdoc.user_password);
	}
	
});


Alloy.Globals.noScreenAlert = false;
Alloy.Globals.actionNotCorrect = null;


socket.on('alertVideoNotCorrect', function(act){
	
	if (Alloy.Globals.noScreenAlert === false) {
		
		alert('The video for the activity ' + act + ' is not suitable. All the human body has to be recorded.');	
	}else{
		
		Alloy.Globals.actionNotCorrect = act;
	}
	
});	


Ti.App.addEventListener('close', function(e){
	
	var file = Titanium.Filesystem.getFile(Titanium.Filesystem.applicationDataDirectory, 'user_info.json'); 
	
	if (file.exists() === true) {
	
		file.deleteFile();
		
	}
});



var actualDateChat = null;
var numberRowsChatView = 0;


function getDate(){
	var currentTime = new Date();
	var hours = ("0" + currentTime.getHours()).slice(-2);
	var minutes = ("0" + currentTime.getMinutes()).slice(-2);
	var seconds = ("0" + currentTime.getSeconds()).slice(-2);
	var day = ("0" + currentTime.getDate()).slice(-2);
	var month = ("0" + (currentTime.getMonth() + 1)).slice(-2);
	var year = currentTime.getFullYear();
	
	
	var time = ''+ year +'-'+ month +'-'+ day +' '+ hours +':'+ minutes +':'+ seconds +'';
	
	return time;
}




