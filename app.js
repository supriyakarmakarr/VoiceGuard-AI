// Compatibility entry point; the current UI loads script.js directly.
if(!document.querySelector('script[src="script.js"]')){const script=document.createElement('script');script.src='script.js';document.head.append(script);}
