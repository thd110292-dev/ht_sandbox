# lpd FDS(Falling Detection Sensing)  - ISK Raw Data 
# ver:0.0.1
# 2021/05/24
# 
# parsing lpd FDS -ISK
# hardware:(Batman-201)ISK IWR6843 ES2.0 
#    Fireware version: I471
# config file: 
# company: Joybien Technologies: www.joybien.com
# author: Zach Chen
#===========================================
# output: V6,V7,V8,V9 Raw data & dataFrame
# v0.0.1 : 2020/06/19 release
#          (1)Output list data
# v0.1.0 : 2021/05/024 
#          (1)Output DataFrame
# v0.1.1 : 2021/05/26
# v0.1.2 : 2021/05/26 bug fix
#
import serial
import time
import struct
import pandas as pd
import numpy as np

class header:
	version = 0
	platform = 0
	timeStamp = 0
	totalPackLen = 0
	frameNumber = 0
	subframeNumber = 0
	chirpMargin = 0
	frameMargin = 0 
	trackProcessTime = 0
	uartSendTime = 0
	numTLVs = 0
	checksum = 0


class LpdFDS:
	
	magicWord =  [b'\x02',b'\x01',b'\x04',b'\x03',b'\x06',b'\x05',b'\x08',b'\x07',b'\0x99']
	port = ""
	hdr = header
	
	# provide csv file dataframe
	# real-time 
	v6_col_names_rt = ['fN','type','range','azimuth','elv','doppler','sx', 'sy', 'sz']
	v7_col_names_rt = ['fN','type','posX','posY','posZ','velX','velY','velZ','accX','accY','accZ','ec0','ec1','ec2','ec3','ec4','ec5','ec6','ec7','ec8','ec9','ec10','ec11','ec12','ec13','ec14','ec15','g','confi','tid']
	v8_col_names_rt = ['fN','type','targetID']
	v9_col_names_rt = ['fN','type','snr','noise']
	# read from file for trace point clouds
	fileName = ''
	v6_col_names = ['time','fN','type','range','azimuth','elv','doppler','sx', 'sy', 'sz']
	v7_col_names = ['time','fN','type','posX','posY','posZ','velX','velY','velZ','accX','accY','accZ','ec0','ec1','ec2','ec3','ec4','ec5','ec6','ec7','ec8','ec9','ec10','ec11','ec12','ec13','ec14','ec15','g','confi','tid']
	v8_col_names = ['time','fN','type','targetID']
	v9_col_names = ['time','fN','type','snr','moise']
	sim_startFN = 0
	sim_stopFN  = 0 
	
	# add for interal use
	tlvLength = 0
	numOfPoints = 0
	# for debug use 
	dbg = False #Packet unpacket Check: True show message 
	sm  = False #Observed StateMachine: True Show message
	plen = 16 
	
	def __init__(self,port):
		self.port = port
		print("(jb)lpd Falling Detection Sensing initial")
		print("(jb)vsersion:v0.1.0")
		print("(jb)For Hardware:Batman-201(ISK)")
		print("(jb)Hardware: IWR-6843")
		print("(jb)Firmware: lpd-I471")
		print("(jb)UART Baud Rate:921600")
		print("Output: V6,V7,V8,V9 data:(RAW)")
		
	def useDebug(self,falseTrue):
		self.dbg = falseTrue
		
	def stateMachine(self,falseTrue):
		self.sm = falseTrue
		
	def getHeader(self):
		return self.hdr
		
		
	def headerShow(self):
		print("******* Header ********") 
		print("Version:     \t%x "%(self.hdr.version))
		print("Platform:    \t%X "%(self.hdr.platform))
		print("TimeStamp:    \t%X "%(self.hdr.timeStamp))
		print("TotalPackLen:\t%d "%(self.hdr.totalPackLen))
		print("PID(frame#): \t%d "%(self.hdr.frameNumber))
		print("subframe#  : \t%d "%(self.hdr.subframeNumber))
		print("Inter-frame Processing Time:\t{:d} us".format(self.hdr.trackProcessTime))
		print("UART Send Time:\t{:d} us".format(self.hdr.uartSendTime))
		print("Inter-chirp Processing Margin:\t{:d} us".format(self.hdr.chirpMargin))
		print("Inter-frame Processing Margin:\t{:d} us".format(self.hdr.frameMargin))
		print("numTLVs:     \t%d "%(self.hdr.numTLVs))
		print("Check Sum   :\t{:x}".format(self.hdr.checksum))
		print("***End Of Header***") 
			
	#for class internal use
	def tlvTypeInfo(self,dtype,count,dShow):
		
		sbyte = 8  #tlvHeader Struct = 8 bytes
		
		dataByte = 0
		pString = ""
		nString = "numOfPoints :"
		stateString = "V6"
		if dtype == 6:
			
			sbyte = 0      #tlvHeader Struct = 8 bytes (type(4)/lenth(4))    0 bytes: not counting
			dataByte= 16    #pointStruct 8bytes:(range(4),azimuth(4),elevation(4),doppler(4))
			pString = "Point Cloud Spherical"
			nString = ""
			stateString = "V6"
		elif dtype == 7:
			
			sbyte = 0	   #tlvHeader Struct = 8 bytes (type(4)/lenth(4))    0 bytes: not counting
			dataByte = 112  #target struct 112 bytes:(tid,posX,posY,posZ,velX,velY,velZ,accX,accY,accZ,ec0,ec1,ec2,ec3,ec4,ec5,ec6,ec7,ec8,ec9,ec10,ec11,ec12,ec13,ec14,ec15,g,confi)  
			pString = "Target Object TLV"
			nString = "numOfObjects:"
			stateString = "V7"
		
		elif dtype == 8:
			sbyte = 0    #tlvHeader Struct = 8 bytes  0 bytes: not counting
			dataByte = 1 #targetID = 1 byte
			pString = "Target Index TLV"
			nString = "numOfIDs"
			stateString = "V8"
		elif dtype == 9:
			
			sbyte = 0    #tlvHeader Struct = 8 bytes  0 bytes: not counting
			dataByte = 4 #snr(2 bytes),noise(2 bytes) = 4 bytes
			pString = "Point Cloud Side Info TLV"
			nString = ""
			stateString = "V9"
		else:
			tlvTypeInfo
			sbyte = 1
			pString = "*** Type Error ***"
			stateString = 'idle'
		 
		retCnt = count - sbyte
		nPoint = retCnt / dataByte
		#dShow = True
		if dShow == True:
			print("-----[{:}] ----".format(pString))
			print("tlv Type({:2d}Bytes):  \t{:d}".format(sbyte,dtype))
			print("tlv length:      \t{:d}".format(count)) 
			print("{:}      \t{:d}".format(nString,int(nPoint)))
			print("value length:    \t{:d}".format(retCnt))  
		
		return stateString, sbyte, dataByte,retCnt, int(nPoint)
		
	def list2df(self,dck,l6,l7,l8,l9):
		ll6 = pd.DataFrame(l6,columns=self.v6_col_names_rt)
		ll7 = pd.DataFrame(l7,columns=self.v7_col_names_rt)
		ll8 = pd.DataFrame(l8,columns=self.v8_col_names_rt)
		ll9 = pd.DataFrame(l9,columns=self.v9_col_names_rt)
		return (dck,ll6,ll7,ll8,ll9)

#
# TLV: Type-Length-Value
# read TLV data
# input:
#     disp: True:print message
#			False: hide printing message
#     df: 
# output:(return parameter)
# (pass_fail, v6, v7, v8, v9)
#  pass_fail: True: Data available    False: Data not available
#  v6: point cloud infomation
#  v7: Target Object information
#  v8: Target Index information
#  v9: Point Cloud Side Information
# 
#
	def tlvRead(self,disp,df = None):
		
		#print("---tlvRead---")
		#ds = dos
		typeList = [6,7,8,9]
		idx = 0
		lstate = 'idle'
		sbuf = b""
		lenCount = 0
		unitByteCount = 0
		dataBytes = 0
		numOfPoints = 0
		tlvCount = 0
		pbyte = 16
		v6 = ([])
		v7 = ([])
		v8 = ([])
		v9 = ([])
		v6df = ([])
		v7df = ([])
		v8df = ([])
		v9df = ([])
		
		while True:
			try:
				ch = self.port.read()
			except:
				return self.list2df(False,v6,v7,v8,v9) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
				
			#print(str(ch))
			if lstate == 'idle':
				#print(self.magicWord)
				if ch == self.magicWord[idx]:
					#print("*** magicWord:"+ "{:02x}".format(ord(ch)) + ":" + str(idx))
					idx += 1
					if idx == 8:
						idx = 0
						lstate = 'header'
						rangeProfile = b""
						sbuf = b""
				else:
					#print("not: magicWord state:")tlvTypeInfo
					idx = 0
					rangeProfile = b""
					return self.list2df(False,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
		
			elif lstate == 'header':
				sbuf += ch
				idx += 1
				if idx == 44:  
					#print("------header-----")
					#print(":".join("{:02x}".format(c) for c in sbuf)) 	 
					#print("len:{:d}".format(len(sbuf))) 
					# [header - Magicword]
					try: 
						(self.hdr.version,self.hdr.platform,
						self.hdr.timeStamp,self.hdr.totalPackLen,
						self.hdr.frameNumber,self.hdr.subframeNumber,
						self.hdr.chirpMargin,self.hdr.frameMargin,self.hdr.trackProcessTime,self.hdr.uartSendTime,
						self.hdr.numTLVs,self.hdr.checksum) = struct.unpack('10I2H', sbuf)
						self.frameNumber = self.hdr.frameNumber
					except:
						if self.dbg == True:
							print("(Header)Improper TLV structure found: ")
						return self.list2df(False,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
					
					if disp == True:  
						self.headerShow()
					
					tlvCount = self.hdr.numTLVs
					#print("tlvCount:{:}".format(tlvCount))
					if self.hdr.numTLVs == 0:
						return self.list2df(True,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (True,v6,v7,v8,v9)
						
					if self.sm == True:
						print("(Header)")
						
					sbuf = b""
					idx = 0
					lstate = 'TL'
					  
				elif idx > 44:
					idx = 0
					lstate = 'idle'
					return self.list2df(False,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
					
			elif lstate == 'TL': #TLV Header type/length
				sbuf += ch
				idx += 1
				if idx == 8:
					#print(":".join("{:02x}".format(c) for c in sbuf))
					try:
						ttype,self.tlvLength = struct.unpack('2I', sbuf)
						if disp == True:
							print("(TL)--tlvNum:{:d}: tlvCount({:d})-------ttype:tlvLength:v{:d}:{:d}".format(self.hdr.numTLVs,tlvCount,ttype,self.tlvLength))
							
						if ttype not in typeList or self.tlvLength > 10000:
							if self.dbg == True:
								print("(TL)Improper TL Length(hex):(T){:d} (L){:x} numTLVs:{:d}".format(ttype,self.tlvLength,self.hdr.numTLVs))
							sbuf = b""
							idx = 0
							lstate = 'idle'
							self.port.flushInput()
							return self.list2df(False,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
							
					except:
						if self.dbg == True:
							print("TL unpack Improper Data Found:")
						self.port.flushInput()
			
						return self.list2df(False,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
					
					lstate ,plen ,dataBytes,lenCount, numOfPoints = self.tlvTypeInfo(ttype,self.tlvLength,disp)
					#if ttype == 6:
					#	print("--pointCloud:((tlvLength({:d})-pointUnit(20)-tlvStruct(8))/8={:d}".format(self.tlvLength,numOfPoints))
					if self.sm == True:
						print("(TL:{:d})=>({:})".format(tlvCount,lstate))
						
					tlvCount -= 1
					idx = 0  
					sbuf = b""
			
					
			elif lstate == 'V6': # count = Total Lentgh - 8
				sbuf += ch
				idx += 1
				#print("v6 dataBytes:{:}  idx:{:} , lenCount:{:}".format(dataBytes,idx,lenCount))
				if (idx%dataBytes == 0):
					try:
						#print(":".join("{:02x}".format(c) for c in sbuf
						#print("V6: sbuf length:{:}".format(len(sbuf)))
						(r,a,e,d) = struct.unpack('4f', sbuf)
						elv = e 
						azi = a 
						dop = d 
						ran = r 
						#print("v6:length:{:}  elv:{:.4f} azimuth:{:.4f} doppler:{:.4f} range:{:.4f}".format(len(v6),elv,azi,dop,ran))
						
						if (df == 'DataFrame'):
							sz  = ran * np.sin(elv)
							sx  = ran * np.cos(elv) * np.sin(azi)
							sy  = ran * np.cos(elv) * np.cos(azi)
							v6df.append((self.hdr.frameNumber,'v6',ran,azi,elv,dop,sx,sy,sz)) # v ok
							 
						else:
							v6.append((ran,azi,elv,dop))
						
						#print("point_cloud_2d.append:[{:d}]".format(len(point_cloud_2d)))
						sbuf = b""
						
					except:
						if self.dbg == True:
							print("(6.1)Improper Type 6 Value structure found: ")
						#return (False,v6,v7,v8)
						#return (False,v6,v7,v8) if (df == None) else self.list2df(False,v6df,v7df,v8df)
						return self.list2df(False,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
					
				if idx == lenCount:
					if disp == True:
						print("v6[{:d}]".format(lenCount))
					idx = 0
					sbuf = b""
					if tlvCount <= 0: # Back to idle
						lstate = 'idle'
						if self.sm == True:
							print("(V6:{:d})=>(idle) :true".format(tlvCount))
						#return (True,v6,v7,v8)
						#return (True,v6,v7,v8) if (df == None) else self.list2df(True,v6df,v7df,v8df)
						return self.list2df(True,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (True,v6,v7,v8,v9)
						
					else: # Go to TL to get others type value
						lstate = 'TL' #'tlTL'
						if self.sm == True:
							print("(V6:{:d})=>(TL)".format(tlvCount))
					
				elif idx > lenCount:
					idx = 0
					sbuf = b""
					lstate = 'idle'
					#return (False,v6,v7,v8)
					#return (False,v6,v7,v8) if (df == None) else self.list2df(False,v6,v7,v8)
					return self.list2df(False,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
					
			 
			elif lstate == 'V9':
				sbuf += ch
				idx += 1
				#print("v9 dataBytes:{:}  idx:{:} , lenCount:{:}".format(dataBytes,idx,lenCount))
				if (idx%dataBytes == 0):
					try:
						#print(":".join("{:02x}".format(c) for c in sbuf))
						(snr,noise) = struct.unpack('2H', sbuf)
						
						#print("v9: snr:{:.4f} noise:{:.4f}".format(snr,noise))
						
						if (df == 'DataFrame'):
							v9df.append((self.hdr.frameNumber,'v9',snr,noise))
						else:
							v9.append((snr,noise))
						
						sbuf = b""
						
					except:
						if self.dbg == True:
							print("(9.1)Improper Type 9 Value structure found: ")
							
						return self.list2df(False,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
					
				if idx == lenCount:
					if disp == True:
						print("v9[{:d}]".format(len(v9)))
					idx = 0
					sbuf = b""
					if tlvCount <= 0: # Back to idle
						lstate = 'idle'
						if self.sm == True:
							print("(V9:{:d})=>(idle) :true".format(tlvCount))
							
						return self.list2df(True,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (True,v6,v7,v8,v9)
						
					else: # Go to TL to get others type value
						lstate = 'TL' #'tlTL'
						if self.sm == True:
							print("(V9:{:d})=>(TL)".format(tlvCount))
					
				elif idx > lenCount:
					idx = 0
					sbuf = b""
					lstate = 'idle'
					#return (False,v6,v7,v8)
					#return (False,v6,v7,v8) if (df == None) else self.list2df(False,v6,v7,v8)
					return self.list2df(False,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
			 
			elif lstate == 'V7':
				sbuf += ch
				idx += 1
				#print("v7 dataBytes:{:}  idx:{:} , lenCount:{:}".format(dataBytes,idx,lenCount))
				 
				if (idx%dataBytes == 0):
					#print("V7:dataBytes({:d}) lenCount({:d}) index:{:d}".format(dataBytes,lenCount,idx))
					try:
						(tid,posX,posY,posZ,velX,velY,velZ,accX,accY,accZ,ec0,ec1,ec2,ec3,ec4,ec5,ec6,ec7,ec8,ec9,ec10,ec11,ec12,ec13,ec14,ec15,g,confi) = struct.unpack('I27f', sbuf)
						if (df == 'DataFrame'): 
							v7df.append((self.hdr.frameNumber,'v7',posX,posY,posZ,velX,velY,velZ,accX,accY,accZ,ec0,ec1,ec2,ec3,ec4,ec5,ec6,ec7,ec8,ec9,ec10,ec11,ec12,ec13,ec14,ec15,g,confi,tid))
						else:
							v7.append((tid,posX,posY,posZ,velX,velY,velZ,accX,accY,accZ,ec0,ec1,ec2,ec3,ec4,ec5,ec6,ec7,ec8,ec9,ec10,ec11,ec12,ec13,ec14,ec15,g,confi))
						
						sbuf = b""
					except:
						if self.dbg == True:
							print("(7)Improper Type 7 Value structure found: ")
						return self.list2df(False,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
						
				if idx >= lenCount:
					if disp == True:
						print("v7[{:d}]".format(len(v7)))
					idx = 0
					sbuf = b""
					if tlvCount == 0:
						lstate = 'idle'
						if self.sm == True:
							print("(V7)=>(idle) :true")
						return self.list2df(True,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (True,v6,v7,v8,v9)
						
					else: # Go to TL to get others type value
						lstate = 'TL'
						idx = 0
						if self.sm == True:
							print("(V7)=>(TL)")

				if idx > lenCount:
					idx = 0 
					lstate = 'idle'
					sbuf = b""
					return self.list2df(False,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
				
			elif lstate == 'V8':
				idx += 1
				v8.append(ord(ch))
				if (df == 'DataFrame'):
					v8df.append((self.hdr.frameNumber,'v8',ord(ch)))
				
				if idx == lenCount:
					if disp == True:
						print("=====V8 End===={:}".format(v8))
					sbuf = b""
					idx = 0
					lstate = 'idle'
					if self.sm == True:
						print("(V8:{:d})=>(idle)".format(tlvCount))
					return self.list2df(True,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (True,v6,v7,v8,v9)
				
				if idx > lenCount:
					sbuf = b""
					idx = 0
					lstate = 'idle'
					return self.list2df(False,v6df,v7df,v8df,v9df) if (df == 'DataFrame') else (False,v6,v7,v8,v9)
					
					
			
			
					
	def v67Simlog(frameNum):
		global sim_startFN,sim_stopFN
		s_fn = frameNum + sim_startFN
		#print("frame number:{:}".format(s_fn))
		v6d = v6sim[v6sim['fN'] == s_fn]
		#v6d =  v6dd[v6dd['doppler'] < 0.0]
		#print(v6d)
		v7d = v7sim[v7sim['fN'] == s_fn]
		chk = 0
		if v6d.count != 0:
			chk = 1
		return (chk,v6d,v7d)
		
	def getRecordData(self,frameNum):
		s_fn = frameNum + self.sim_startFN
		#print("frame number:{:}".format(s_fn))
		v6d = self.v6simo[self.v6simo['fN'] == s_fn]
		#v6d =  v6dd[v6dd['doppler'] < 0.0]
		#print(v6d)
		v7d = self.v7simo[self.v7simo['fN'] == s_fn]
		v8d = self.v8simo[self.v8simo['fN'] == s_fn]
		chk = 0
		if v6d.count != 0:
			chk = 1
		return (chk,v6d,v7d,v8d)
		
	def readFile(self,fileName):
		#fileName = "pc32021-03-19-10-02-17.csv"  
		#df = pd.read_csv(fileName, error_bad_lines=False, warn_bad_lines=False) 
		self.fileName = fileName 
		#          ['time','fN','type','elv','azimuth','range' ,'doppler','snr','sx', 'sy', 'sz']
		df = pd.read_csv(self.fileName, names = self.v6_col_names, skiprows = [0,11,12]) 
		df.dropna()
		#print("------------------- df --------------------shape:{:}".format(df.shape))
		print(df.info())
		print(df.info(memory_usage="deep")) 
		
		v6simOri = df[(df.type == 'v6')]
		#print("-------------------v6sim------------:{:}".format(v6simOri.shape))
		#self.v6simo = v6simOri.loc[:,['fN','elv','azimuth','range','doppler','snr']]
		self.v6simo = v6simOri.loc[:,['fN','type','elv','azimuth','range' ,'doppler','snr','sx', 'sy', 'sz']]
		self.v6simo['elv'] = self.v6simo['elv'].astype(float, errors = 'raise') 
		
		df7 = pd.read_csv(self.fileName, names = self.v7_col_names, skiprows = [0])  
		
		v7simc = df7[df7['type'] == 'v7']
		self.v7simo  = v7simc.loc[:,['fN','posX','posY','posZ','velX','velY','velZ','accX','accY','accZ','ec0','ec1','ec2','ec3','ec4','ec5','ec6','ec7','ec8','ec9','ec10','ec11','ec12','ec13','ec14','ec15','g','confi','tid']]
		self.sim_startFN = df['fN'].values[0]
		self.sim_stopFN  = df['fN'].values[-1]
		'''
		print("---------start frame number---------:{:}".format(self.sim_startFN))
		print("---------stop  frame number---------:{:}".format(self.sim_stopFN))
		print(self.v7simo)
		print("-----v8sim data lib-----")
		'''
		v8simc = df[df['type'] == 'v8']
		self.v8simo  = v8simc.loc[:,['fN','type','elv']]
		self.v8simo.columns = ['fN','type','targetID']
		#print(self.v8simo)
		return (self.v6simo,self.v7simo,self.v8simo)


