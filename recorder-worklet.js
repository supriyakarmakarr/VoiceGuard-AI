class VoiceGuardRecorder extends AudioWorkletProcessor {
 constructor(){super();this.buffer=new Float32Array(2048);this.index=0;}
 process(inputs){const channels=inputs[0];if(channels?.length)for(let i=0;i<channels[0].length;i++){let sum=0;for(const channel of channels)sum+=channel[i]||0;this.buffer[this.index++]=sum/channels.length;if(this.index===this.buffer.length){this.port.postMessage(this.buffer.buffer,[this.buffer.buffer]);this.buffer=new Float32Array(2048);this.index=0;}}return true;}
}
registerProcessor('voiceguard-recorder',VoiceGuardRecorder);
