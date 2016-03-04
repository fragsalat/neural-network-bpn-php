<?php
/**
 * Created by IntelliJ IDEA.
 * User: schlage
 * Date: 15.01.16
 * Time: 12:56
 */

include_once 'Network.php';

function createSample($char, $angle, $font) {
	// Create empty image and base colors
	$img = imagecreatetruecolor(28, 28);
	$black = imagecolorallocate($img, 0, 0, 0);
	$white = imagecolorallocate($img, 255, 255, 255);
	// Fill background with white, draw character and ensure the image is grayscaled
	imagefill($img, 0, 0, $white);
	imagettftext($img, 18, $angle, 2, 20, $black, $font, $char);
	imagefilter($img, IMG_FILTER_GRAYSCALE);

//	header('Content-Type: image/jpeg');
//	imagejpeg($img);
//	exit;
	// Get binary of ascii value
	$binary = decbin(ord($char));
	$binary = str_repeat('0', 7 - strlen($binary)) . $binary;
	$desired = array_map('intval', str_split($binary));
	// Prepare sample
	$sample = [$char, $desired, []];
	// Get sample data
	for ($x = 0, $w = imagesx($img); $x < $w; $x++) {
		for ($y = 0, $h = imagesy($img); $y < $h; $y++) {
			$sample[2][] = 1 - imagecolorat($img, $x, $y) / 16777215;
		}
	}

	return $sample;
}

$key = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
$fonts = [
	'/usr/share/fonts/truetype/droid/DroidSans.ttf',
	'/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
	'/usr/share/fonts/truetype/freefont/FreeMono.ttf',
	'/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf',
	'/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
];
$samples = [];
for ($k = 100; $k--;) {
	for ($i = strlen($key); $i--;) {
		foreach ($fonts as $font) {
			$samples[] = createSample($key[$i], rand(-10, 10), $font);
		}
	}
}

$network = new Network(count($samples[0][2]), 60, 7);
$network->load('/tmp/network4');
shuffle($samples);
$chunks = array_chunk($samples, 1000);
$train = [];

$s = $samples[0];
$out = $network->run($s[2]);
print_r($out);
print_r($s[1]);
print_r($s[0]);
exit;

for ($epoch = 0; $epoch < 100; $epoch++) {
	$train = array_shift($chunks);
	$accuracy = 0;
	$runs = 0;
	$start = microtime(true);

	echo 'Start train ' . count($train) . ' in epoch ' . $epoch . PHP_EOL;

	for ($i = 10; $i--;) {
		foreach ($train as $sample) {
			$accuracy += $network->train($sample[2], $sample[1]);
			$runs++;
		}
	}

	echo 'Epoch ' . $epoch . ' accuracy ' . ($accuracy / ($runs ?: 1)) . ' in ' . (microtime(true) - $start) . PHP_EOL;
	$start = microtime(true);

	//$train = [];
	$matches = 0;
	foreach ($train as $s => $sample) {
		$out = $network->run($sample[2]);
		$binary = '';
		foreach ($out as $bit) {
			$binary .= $bit > .8 ? 1 : 0;
		}

		$char = chr(bindec($binary));
		if ($char == $sample[0]) {
			$matches++;
		}
	}

	echo "Matched " . $matches . " of " . count($train) . ' in ' . (microtime(true) - $start) . PHP_EOL;

	$network->save('/tmp/network4');
}

/*
for ($s = count($samples); $s--;) {
	$out = $network->run($samples[$s][1]);
	$max = max($out);
	if ($max > .9) {
		for ($o = count($out); $o--;) {
			if ($out[$o] == $max) {
				echo 'Sample ' . $s . ' is a ' . ($o + 1) . PHP_EOL;
			}
		}
	}
	else {
		echo 'Sample ' . $s . ' is unknown' . implode($out, ', ') . PHP_EOL;
	}
}*/