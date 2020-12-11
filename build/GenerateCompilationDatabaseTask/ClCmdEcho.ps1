# Output the arguments
#
# NOTE: These will have been processed by Powershell first,
# so comma separated items will have been converted to arrays.
# 
$cmdLine = ''
for ($i=0; $i -lt $args.Count; $i+=1)
{
	if ($cmdLine -ne '')
	{
		$cmdLine += ' '
	}

	if ($args[$i].GetType().IsArray)
	{
		$cmdLine += ($args[$i] -join ',')
	}
	else
	{
		$cmdLine += $args[$i]
	}
}
echo "$cmdLine"
