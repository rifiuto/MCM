$0!~/^# In/{
	len=length
	for(i=1; i<=len;i++){
		if(substr($0,i,1)<="\00" || substr($0,i,1)>"\177"){
			# use blank to substitue this, otherwise has index issue
			gsub(substr($0,i,1)," ");
		}
	}
	print
}
