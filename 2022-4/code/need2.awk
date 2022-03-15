$0!~/^# In/{
	len=length
	gsub("，", ",")
	for(i=1; i<=len;i++){
		char=substr($0,i,1);
		if(char>"\343" && char<"\352"){
			# use blank to substitue this, otherwise has index issue
			gsub(char," ");
		}
	}
	print
}
